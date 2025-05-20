import os

import litellm
import torch
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

from akd.tools.fact_reasoner.fm_factual.utils import (
    DEFAULT_PROMPT_BEGIN,
    DEFAULT_PROMPT_END,
    HF_MODELS,
    RITS_MODELS,
)

GPU = torch.cuda.is_available()
DEVICE = GPU * "cuda" + (not GPU) * "cpu"


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class LLMHandler:
    def __init__(
        self,
        model: str,
        RITS: bool = True,
        dtype="auto",
        **default_kwargs,
    ):
        """
        Initializes the LLM handler.

        :param model_id: Model name or path.
        :param RITS: If True, use RITS model; if False, load using vLLM.
        :param default_kwargs: Default parameters (e.g., temperature, max_tokens) to pass to completion calls.
        """
        self.RITS = RITS
        self.default_kwargs = default_kwargs  # Store common parameters for completions

        if not self.RITS:
            self.HF_model_info = HF_MODELS[model]
            self.model_id = self.HF_model_info.get("model_id", None)
            assert self.model_id is not None
            print(f"Loading local model with vLLM: {self.model_id}...")
            self.llm = LLM(
                model=self.model_id,
                device=DEVICE,
                dtype=dtype,
            )  # Load model using vLLM
        else:
            load_dotenv(override=True)
            self.RITS_API_KEY = os.getenv("OPENAI_API_KEY")

            self.rits_model_info = RITS_MODELS[model]

            self.prompt_template = self.rits_model_info.get("prompt_template", None)
            self.max_new_tokens = self.rits_model_info.get("max_new_tokens", None)
            self.api_base = self.rits_model_info.get("api_base", None)
            self.model_id = self.rits_model_info.get("model_id", None)
            self.prompt_begin = self.rits_model_info.get(
                "prompt_begin",
                DEFAULT_PROMPT_BEGIN,
            )
            self.prompt_end = self.rits_model_info.get("prompt_end", DEFAULT_PROMPT_END)
            assert (
                self.prompt_template is not None
                and self.max_new_tokens is not None
                and self.api_base is not None
                and self.model_id is not None
            )

    def completion(self, prompt, **kwargs):
        """
        Generate a response using the RITS API (if RITS=True) or the local model.

        :param message: The prompt.
        :param kwargs: Additional parameters for completion (overrides defaults).
        """
        return self._call_model(prompt, **kwargs)

    def batch_completion(self, prompts, **kwargs):
        """
        Generate responses in batch using the RITS API (if RITS=True) or the local model.

        :param prompts: List of prompts.
        :param kwargs: Additional parameters for batch completion.
        """
        return self._call_model(prompts, **kwargs)

    def _call_model(self, prompts, **kwargs):
        """
        Handles both single and batch generation.

        :param prompts: A single string or a list of strings.
        :param kwargs: Additional parameters.
        """
        params = {
            "temperature": 0,
            "seed": 42,
            # the two above are overwritten if passed
            # as kwargs
            **self.default_kwargs,
            **kwargs,
        }  # Merge defaults with provided params

        if self.RITS:
            # Ensure we always send a list to batch_completion
            if isinstance(prompts, str):
                return litellm.completion(
                    model="gpt-4o-mini",
                    api_base="https://api.openai.com/v1",
                    messages=[
                        {"role": "user", "content": prompts},
                    ],  # Wrap prompt for compatibility
                    api_key=self.RITS_API_KEY,
                    # extra_headers={
                    #     "RITS_API_KEY": self.RITS_API_KEY
                    # },
                    **params,
                )
            return litellm.batch_completion(
                model="gpt-4o-mini",
                api_base="https://api.openai.com/v1",
                messages=[
                    [{"role": "user", "content": p}] for p in prompts
                ],  # Wrap each prompt
                api_key=self.RITS_API_KEY,
                # extra_headers={
                #     "RITS_API_KEY": self.RITS_API_KEY
                # },
                **params,
            )
        else:
            # Ensure prompts is always a list for vLLM
            if isinstance(prompts, str):
                prompts = [prompts]

            sampling_params = SamplingParams(**params)
            outputs = self.llm.generate(prompts, sampling_params)

            # print("\n=== FULL OUTPUT STRUCTURE ===\n")
            # self.recursive_print(outputs)

            # import pickle
            # with open("saved_vllm_response.pkl",'wb') as f:
            #    pickle.dump(outputs,f)

            # Convert vLLM outputs to match litellm format
            responses = [self.transform_vllm_response(output) for output in outputs]

            return responses if len(prompts) > 1 else responses[0]
            # return [output.outputs[0].text for output in outputs] #TODO: make output consistent with that of RITS

    def transform_vllm_response(self, response_obj):

        output_obj = response_obj.outputs[0]

        # Extract the generated text
        text = output_obj.text

        # Convert logprobs into the expected structure
        logprobs = []
        for token_dict in output_obj.logprobs:
            best_token_id = max(
                token_dict,
                key=lambda k: token_dict[k].rank,
            )  # Select top-ranked token
            logprobs.append(
                {
                    "logprob": token_dict[best_token_id].logprob,
                    "decoded_token": token_dict[best_token_id].decoded_token,
                },
            )

        # Create the transformed response
        transformed_response = dotdict(
            {
                "choices": [
                    dotdict(
                        {
                            "message": dotdict({"content": text}),
                            "logprobs": {"content": logprobs},
                        },
                    ),
                ],
            },
        )

        return transformed_response

    def recursive_print(self, obj, indent=0):
        """Recursively print objects, lists, and dicts for deep inspection."""
        prefix = "  " * indent  # Indentation for readability

        if isinstance(obj, list):
            print(f"{prefix}[")
            for item in obj:
                self.recursive_print(item, indent + 1)
            print(f"{prefix}]")
        elif isinstance(obj, dict):
            print(f"{prefix}{{")
            for key, value in obj.items():
                print(f"{prefix}  {key}: ", end="")
                self.recursive_print(value, indent + 1)
            print(f"{prefix}}}")
        elif hasattr(obj, "__dict__"):  # Print class attributes
            print(f"{prefix}{obj.__class__.__name__}(")
            for key, value in vars(obj).items():
                print(f"{prefix}  {key}: ", end="")
                self.recursive_print(value, indent + 1)
            print(f"{prefix})")
        else:
            print(f"{prefix}{repr(obj)}")  # Print basic values


if __name__ == "__main__":

    """
    Test to compare RITS (litellm) and local (vLLM) outputs.
    """
    test_prompt = "What is the capital of France?"

    # RITS (litellm) API
    remote_handler = LLMHandler(
        model="llama-3.1-70b-instruct",
        RITS=True,
    )

    remote_response = remote_handler.completion(
        test_prompt,
        logprobs=True,
        seed=12345,
    )
    print("\nREMOTE RESPONSE:")
    print(remote_response)

    # Local (vLLM) - Using a small model for testing
    """
    local_handler = LLMHandler(
        model="mixtral-8x7b-instruct",
        RITS=False,
        dtype="half",
        logprobs=1
    )
    """

    local_handler = LLMHandler(
        model="facebook/opt-350m",
        RITS=False,
        logprobs=1,
    )

    local_response = local_handler.completion(test_prompt)
    print("\nLOCAL RESPONSE:")
    print(local_response)

    # Ensure the response has 'choices' attribute
    assert hasattr(remote_response, "choices"), "Remote response missing 'choices'"
    assert hasattr(local_response, "choices"), "Local response missing 'choices'"

    # Ensure 'choices' is a list
    assert isinstance(
        remote_response.choices,
        list,
    ), "'choices' should be a list in remote response"
    assert isinstance(
        local_response.choices,
        list,
    ), "'choices' should be a list in local response"
    assert len(remote_response.choices) > 0, "Remote response 'choices' is empty"
    assert len(local_response.choices) > 0, "Local response 'choices' is empty"

    # Ensure the first choice has 'message' and 'logprobs'
    assert hasattr(
        remote_response.choices[0],
        "message",
    ), "Remote response missing 'message' in choices[0]"
    assert hasattr(
        local_response.choices[0],
        "message",
    ), "Local response missing 'message' in choices[0]"
    assert hasattr(
        remote_response.choices[0].message,
        "content",
    ), "Remote response missing 'content' in message"
    assert hasattr(
        local_response.choices[0].message,
        "content",
    ), "Local response missing 'content' in message"

    assert hasattr(
        remote_response.choices[0],
        "logprobs",
    ), "Remote response missing 'logprobs' in choices[0]"
    assert hasattr(
        local_response.choices[0],
        "logprobs",
    ), "Local response missing 'logprobs' in choices[0]"
    assert isinstance(
        remote_response.choices[0].logprobs,
        dict,
    ), "'logprobs' should be a dictionary in remote response"
    assert isinstance(
        local_response.choices[0].logprobs,
        dict,
    ), "'logprobs' should be a dictionary in local response"
    assert (
        "content" in remote_response.choices[0].logprobs
    ), "Remote response missing 'content' in logprobs"
    assert (
        "content" in local_response.choices[0].logprobs
    ), "Local response missing 'content' in logprobs"

    print("\n✅ Test passed: Both remote and local responses follow the same structure.")
