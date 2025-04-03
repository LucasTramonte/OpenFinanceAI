import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def generate_responses(query: str, relevant_images: list):
    """Generate responses based on the top-k relevant pages."""
    try:
        # Load the Qwen2.5-VL model and processor
        gen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16
        ).cuda().eval()
        max_pixels = 512 * 28 * 28
        gen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=max_pixels)

        # Prepare the prompt with the query
        PROMPT = f"Use the following pages to answer the query:\n{query}\n"

        # Prepare the messages with multiple images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path} for image_path in relevant_images
                ] + [{"type": "text", "text": PROMPT}],
            }
        ]
        text = gen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = gen_processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda")

        # Generate a unique response
        torch.cuda.empty_cache()
        generated_ids = gen_model.generate(**inputs, max_new_tokens=150)
        output_text = gen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Save the generated response
        response_path = os.path.join("..", "Assets", "output", "generated_responses.txt")
        with open(response_path, "w") as f:
            f.write(output_text)

        return output_text

    except Exception as e:
        raise RuntimeError(f"An error occurred during response generation: {e}")