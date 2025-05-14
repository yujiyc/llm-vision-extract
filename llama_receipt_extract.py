import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Path to the directory containing preprocessed images of the delivery receipts
image_dir = "/home/yuji/llama/preprocessed-imgs"

# Load the LLaMA-3.2 Vision model and processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Prompt for the model, instructing it to extract text from color-coded regions
prompt = """
You are analyzing an image of a delivery receipt. This image has been pre-processed with visual detection, and four specific regions are highlighted with colored boxes. Each color corresponds to a key piece of information to be extracted as text.

The color-to-information mapping is:
- Red box: **Date** (the date the item was received)
- Green box: **Stamp** (stamp identification)
- Blue box: **Recipient’s signature** (name or signature of the person who received it)
- Yellow box: **Invoice number and series** (fiscal document ID)

Your task is to:
1. Read and extract the text from each of these colored boxes.
2. Organize the extracted information into a JSON object with the following fields:
   - `date`
   - `signature`
   - `stamp`
   - `invoice_number_and_series`

Rules:
- If any information is illegible or missing, you must use the value `null` for that field.
- Your response **must be the JSON object only**, without any comments, explanations, or extra formatting.

Expected output format (example structure only):
```json
{
  "date": "[DATE]",
  "signature": "[SIGNATURE]",
  "stamp": "[STAMP]",
  "invoice_number_and_series": "[INVOICE_AND_SERIES]"
}
"""
# Create the output directory if it doesn't exist
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Loop over all image files in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_dir, filename)
        try:
            # Load the image and convert to RGB format
            image = Image.open(image_path).convert("RGB")

            # Construct message for the multimodal model (image + text prompt)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt.strip()},
                    ],
                }
            ]
            
            # Apply chat template and prepare model inputs
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(image, input_text, return_tensors="pt").to(model.device)

            # Generate model output
            output = model.generate(**inputs, max_new_tokens=1024)
            decoded_output = processor.decode(output[0], skip_special_tokens=True)

            # Save output as text file (one per image)
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_dir, f"{base_name}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(decoded_output)

            print(f"[✓] Output saved for: {filename}")

        except Exception as e:
            # Handle any errors during processing
            print(f"[X] Error processing {filename}: {e}")