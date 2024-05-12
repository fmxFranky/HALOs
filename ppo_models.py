import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from models import PreTrainedModelWrapper, ValueHead


class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, pretrained_model, *args, **kwargs):
        super().__init__(pretrained_model)
        v_head_kwargs, other_kwargs = self._split_kwargs(kwargs)

        if not any(
            hasattr(self.pretrained_model, attribute)
            for attribute in self.lm_head_namings
        ):
            raise ValueError(
                "The model does not have a language model head, please use a model that has one."
            )

        self.v_head = ValueHead(
            self.pretrained_model.config, freeze_llm_backbone=False, **v_head_kwargs
        )
        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":

            def weights_init(m):
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=initializer_range)
                    m.bias.data.zero_()

            self.summary.apply(weights_init)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = (
            True  # this had already been set in the LORA / PEFT examples
        )
        kwargs["past_key_values"] = past_key_values
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        # force upcast in fp32 if logits are in half-precision
        if last_hidden_state.dtype != torch.float32:
            last_hidden_state = last_hidden_state.float()
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        scores = self.v_head(last_hidden_state)

        end_index = torch.cat([m.nonzero()[-1] for m in attention_mask])  # size = (B,)
        end_scores = torch.gather(  # size = (B, 1, D)
            scores,
            dim=1,
            index=(
                end_index.to(scores.device)
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .expand(-1, -1, scores.size(-1))
            ),
        )
        return scores, end_scores.squeeze(dim=1)

    def state_dict(self, *args, **kwargs):
        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, "v_head", self.v_head)
        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def post_init(self, state_dict):
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

