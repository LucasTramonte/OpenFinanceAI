import os
import torch
import logging
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Setup logging
logger = logging.getLogger(__name__)

def generate_responses(query: str, relevant_images: list):
    """Generate responses based on the top-k relevant images."""
    try:
        # Load the model & processor
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        logger.info(f"Loading model: {model_name}")

        gen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).cuda().eval()

        gen_processor = AutoProcessor.from_pretrained(model_name, max_pixels=512 * 28 * 28)

        # Prepare the prompt
        prompt = f"Use the following pages to answer the query:\n{query}\n"
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in relevant_images] + 
                           [{"type": "text", "text": prompt}],
            }
        ]

        text = gen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare input tensors
        inputs = gen_processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda")

        # Free memory before inference
        torch.cuda.empty_cache()

        # Generate response
        logger.info("Generating response...")
        generated_ids = gen_model.generate(**inputs, max_new_tokens=150)
        output_text = gen_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Save response
        response_path = os.path.join("..", "Assets", "output", "generated_responses.txt")
        os.makedirs(os.path.dirname(response_path), exist_ok=True)  # Ensure directory exists
        with open(response_path, "w") as f:
            f.write(output_text)

        logger.info("Response generation complete.")
        return output_text

    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise RuntimeError(f"An error occurred during response generation: {e}")
