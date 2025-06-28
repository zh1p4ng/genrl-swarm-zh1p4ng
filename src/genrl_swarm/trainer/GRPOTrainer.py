import os, gc
from collections import defaultdict
from typing import Any, List, Dict, Tuple

import torch
import torch.utils.data
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from genrl_swarm.trainer import TrainerModule
from genrl_swarm.rewards import RewardManager
from genrl_swarm.data import DataManager
from genrl_swarm.state import GameState
from genrl_swarm.logging_utils.ml_logger import LoggerMixin
from trl.trainer.grpo_config import GRPOConfig
from trl.models import create_reference_model
from trl.data_utils import apply_chat_template
from reasoning_gym.utils import SYSTEM_PROMPTS

class GRPOTrainerModule(TrainerModule, LoggerMixin):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    Implements the TrainerModule interface defined in base_trainer.py.
    """
    
    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the GRPO trainer module.
        
        Args:
            models: List containing the model to be trained.
            **kwargs: Additional arguments for configuration.
        """
        # Extract model and reward functions
        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")
        
        self.model = models[0] #TODO(Discuss): How to settup multiple models here? Should be tethered to agent index that'll be given by gamestate. Maybe loop here and add a lil model ID datum to the gamestate?
        
        # Configuration parameters
        config = kwargs.get("config", None)
        self.args = config if isinstance(config, GRPOConfig) else GRPOConfig(config) if config else GRPOConfig()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        # Tokenizers
        self.processing_class = kwargs.get("processing_class", None)
        
        # Additional parameters
        self.callbacks = kwargs.get("callbacks", [])
        self.save_dir = kwargs.get("log_dir", "./outputs")
        self.global_step = 0
        self.num_generations = kwargs.get("num_generations", 2)
        assert self.num_generations > 1, f"For GRPO training, number of generations must be > 1, got {self.num_generations}"
        self.epsilon = kwargs.get("epsilon", 0.2)
        self.epsilon_high = kwargs.get("epsilon_high", None)
        self.beta = kwargs.get("beta", 0.0)
        self.judge_base_url = kwargs.get("judge_base_url", None)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Initialize core components
        self._initialize_model()
        self._initialize_tokenizers()
        self._initialize_metrics()
        self._initialize_generation_config()
        self.init_tracker(self.save_dir, log_with=kwargs.get("log_with", None))
    
    def _initialize_model(self):
        """Initialize the model and reference model."""
        # # Set up model hyperparameters
        # self.beta = self.args.beta
        self.model = self.model.to(self.device)

        # Reference model setup
        if self.beta == 0.0:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.model).to(self.model.device)
    
    def _initialize_tokenizers(self):
        """Initialize tokenizers for the model and reward models."""
        if self.processing_class is None:
            self.processing_class = AutoTokenizer.from_pretrained(
                self.model.config._name_or_path, 
                padding_side="left"
            )
    
    def _initialize_metrics(self):
        """Initialize metrics tracking for training and evaluation."""
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
    
    def _initialize_generation_config(self):
        # Set generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            bos_token_id=self.processing_class.bos_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            min_p=self.args.min_p,
            repetition_penalty=self.args.repetition_penalty
        )
        
    def _process_inputs(self, inputs, with_template=True, for_training=False):
        if hasattr(inputs, 'to_dict'):
            inputs = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            inputs = [inputs]

        if with_template: #Pick up here!!!! Remove the for generation arg and instead unflatten the templated prompts to get back tensor of shape [batch size, completions, tokens]
            if for_training:
                templated_prompts = []
                for item in inputs:
                    for _ in range(self.num_generations):
                        templated_prompts.append(apply_chat_template(item, self.processing_class)['prompt'])
            else:
                templated_prompts = [apply_chat_template(item, self.processing_class)['prompt'] for item in inputs]

        else:
            if for_training:
                templated_prompts = []
                for generations in inputs:
                    for output in generations:
                        templated_prompts.append(output) #[item[0] for item in inputs]
            else:
                templated_prompts = [item[0] for item in inputs]

        input_tokens = self.processing_class(
            text=templated_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        return input_tokens

    def generate(self, inputs: Any, return_completion_ids:bool = False, stage=0) -> Any:
        """
        Generate outputs from the model for the given inputs.
        
        Args:
            inputs: Input data for generation
            return_completion_ids: Whether to return completion IDs along with text
            stage: Current stage (0, 1, or 2) for proper output formatting
            
        Returns:
            Generated outputs in the format expected by the next stage
        """   
        input_tokens = self._process_inputs(inputs)
        rollout, rollout_ids = [], [] #TODO: Revisit this for getting a larger number of completions. Super hacky and ugly currently.
        for _ in range(self.num_generations):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_tokens.input_ids.to(self.model.device),
                    attention_mask=input_tokens.attention_mask.to(self.model.device),
                    generation_config=self.generation_config,
                )
                
            # Extract completions (i.e., removes prompt part)
            prompt_length = input_tokens.input_ids.size(1)
            completion_ids = outputs[:, prompt_length:]

            completions = self.processing_class.batch_decode(
                completion_ids, 
                skip_special_tokens=True
            )

            if len(rollout) == 0:
                rollout = [[comp] for comp in completions]
                if return_completion_ids:
                    rollout_ids = [[comp] for comp in completion_ids]
            else:
                for idx, comp in enumerate(completions):
                    rollout[idx].append(comp)
                    if return_completion_ids:
                        rollout_ids[idx].append(completion_ids[idx])
        if return_completion_ids:
            return rollout, rollout_ids
        else:
            return rollout
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """Get the per-token log probabilities for the input tokens.
        
        Args:
            model: The model to compute log probabilities for.
            input_ids: The input token IDs.
            attention_mask: The attention mask.
            logits_to_keep: The number of logits to keep.
            
        Returns:
            The per-token log probabilities.
        """
        model = model.to(input_ids.device) #this shouldn't be needed
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred


        loss_mask = attention_mask[:, -logits_to_keep:].to(dtype=logits.dtype).contiguous()
        labels = input_ids[:, -logits_to_keep:].contiguous()
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        logits = logits[:, -logits_to_keep:].contiguous()
        # Divide logits by sampling temperature.
        logits = logits / self.args.temperature
        logits_shape = logits.shape
        token_log_probs = - torch.nn.functional.cross_entropy(
            logits.view(-1, logits_shape[-1]),
            labels.view(-1),
            reduction='none',
        ).view(logits_shape[0], logits_shape[1])
        token_log_probs = token_log_probs * loss_mask + (1.0 - loss_mask) * torch.finfo(logits.dtype).min
        return token_log_probs  # compute logprobs for the input tokens

    def compute_loss(self, model, inputs, num_items_in_batch=1, mode='train', return_metrics=False):
        """Compute the GRPO loss.
        
        Args:
            model: The model to compute the loss for.
            inputs: The inputs containing prompt_ids, prompt_mask, completion_ids, completion_mask,
                    old_per_token_logps, ref_per_token_logps, and advantages.
            
        Returns:
            The loss value and metrics.
        """
        
        # Extract inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        
        # Concatenate prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(self.model.device)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        # Compute per-token log probabilities
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        
        # Compute KL divergence between model and reference model if beta > 0
        if self.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, input_ids, attention_mask, logits_to_keep)
            else:
                ref_per_token_logps = per_token_logps.clone()

            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
        
        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        old_per_token_logps = inputs["old_per_token_logps"] if self.args.num_iterations > 1 else per_token_logps.detach()
        
        # Calculate ratios and loss terms
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon_high if self.epsilon_high is not None else self.epsilon)
        advantages =  advantages.unsqueeze(dim=-1)

        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # Add KL penalty if beta > 0
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        # Final loss calculation
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(mean_kl.item())
        
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())

        # return for tensorboard
        metrics = {
            "loss": loss.item(),
            "kl": mean_kl.item() if self.beta != 0.0 else None,
            "clip_ratio": clip_ratio.item(),
            }

        if return_metrics:
            return loss, metrics
        else:
            return loss
 
    def train(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager) -> None:
        """
        Train the model using the given game state and reward manager.
        
        Args:
            game_state: The current game state.
            reward_manager: The reward manager to use for computing rewards.
        """
        self.model.train()
        global_step = self.global_step
        for stage in range(state.stage):
            global_step = self.step(stage, state, data_manager, reward_manager, global_step)
        self.global_step = global_step
        self.model.eval()

    def step(self, stage: int, state: GameState, data_manager: DataManager, reward_manager: RewardManager, global_step: int) -> int:
        global_step += 1        

        #Prepare stage's inputs
        stage_inputs = state.get_stage_state(stage) # Fetches the current world state for all agents
        stage_inputs, index_mapping = data_manager.prepare_input(stage_inputs, stage) # Maps game tree states to model ingestable inputs
        assert stage_inputs is not None, f"No inputs found for stage {stage}"
        #Unflatten stage's outputs
        stage_actions = state.get_stage_actions(stage)
        stage_outputs = [stage_actions[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]] for idx, _ in enumerate(index_mapping)]
        assert stage_outputs is not None, f"No outputs found for stage {stage}"

        model_inputs = {}
        processed_inputs = self._process_inputs(stage_inputs, for_training=True)
        model_inputs['prompt_ids'], model_inputs['prompt_mask'] = processed_inputs.input_ids.to(self.model.device), processed_inputs.attention_mask.to(self.model.device)
        processed_outputs = self._process_inputs(stage_outputs, with_template=False, for_training=True)
        model_inputs['completion_ids'], model_inputs['completion_mask'] = processed_outputs.input_ids.to(self.model.device), processed_outputs.attention_mask.to(self.model.device)

        rewards = reward_manager[stage]
        rewards = [rewards[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]] for idx, _ in enumerate(index_mapping)]
        assert rewards is not None, f"No rewards found for stage {stage}"
        rewards = torch.tensor(rewards)

        with torch.no_grad():
            advantages = (rewards - rewards.mean(dim=1, keepdim=True)) 
            if rewards.shape[1] > 1:
                advantages /= (rewards.std(dim=1, keepdim=True) + 1e-8)
        advantages = torch.flatten(advantages).to(self.model.device)
        
        model_inputs["advantages"] = advantages.squeeze(dim=-1)
        model_inputs["old_per_token_logps"] = None

        with torch.amp.autocast(device_type="cuda", enabled=self.args.fp16):
            loss = self.compute_loss(self.model, model_inputs)
        
        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()
      
        metrics = {'train/loss': loss.cpu().mean().item()} 
        metrics.update({'train/rewards': rewards.cpu().mean().item()})
        self.log(metrics, global_step)

        self.cleanup_step()

        return global_step

    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        import requests
        from genrl_swarm.logging_utils.global_defs import get_logger
        base_url = self.judge_base_url
        if base_url:
            try:
                request_data = {'user_id': state.peer_id}
                response = requests.post(f"{base_url}/request-question/", json=request_data)

                if response.status_code == 200:
                    result = response.json()
                    get_logger().debug(f'recieved question: {result["question"]}')
                else:
                    get_logger().debug(f'Failed to recieve question: {response.status_code}')
                    return
                
                prompt = [
                        {"role": "system", "content": SYSTEM_PROMPTS['default']},
                        {"role": "user", "content": result["question"]}
                    ]
                input_ids = self.processing_class.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                input_ids = input_ids.to(self.model.device)
                outputs = self.model.generate(input_ids, max_new_tokens=512)
                answer = self.processing_class.decode(outputs[0], skip_special_tokens=True)
                session_id = result["session_id"]
                submission_data = {
                    "session_id": session_id,
                    "round_number": state.round,
                    "user_answer": answer
                }
                response = requests.post(
                    f"{base_url}/submit-answer/",
                    json=submission_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    get_logger().debug(f"Score: {result['score']}")
                    return 
                else:
                    get_logger().debug(f"Failed to submit answer: {response.status_code}")
                    return
            except Exception as e:
                get_logger().debug(f"Failed to evaluate: {e}")
                return
        else:
            return

        
    def save(self, save_dir: str) -> None:
        """
        Save the model and trainer state to the given directory.
        
        Args:
            save_dir: The directory to save to.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.trainer.save_model(save_dir)
        
        # Save additional state
        torch.save({
            "metrics": self._metrics,
            "total_train_tokens": self._total_train_tokens,
            "generation_config": self.generation_config,
        }, os.path.join(save_dir, "trainer_state.pt"))
    
    @classmethod
    def load(cls, load_dir: str) -> 'GRPOTrainerModule':
        """
        Load a trainer module from the given directory.
        
        Args:
            load_dir: The directory to load from.
            
        Returns:
            The loaded trainer module.
        """
        # Load model
        model = AutoModelForCausalLM.from_pretrained(load_dir)
        
        # Create trainer instance
        trainer = cls([model])
        
        # Load additional state
        trainer_state = torch.load(os.path.join(load_dir, "trainer_state.pt"))
        trainer._metrics = trainer_state["metrics"]
        trainer._total_train_tokens = trainer_state["total_train_tokens"]
        trainer.generation_config = trainer_state["generation_config"]
        
        return trainer

    def cleanup_step(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def cleanup(self):
        self.cleanup_trackers()