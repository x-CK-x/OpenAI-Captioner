import openai
import requests
from PIL import Image, ImageOps, ImageFilter
from io import BytesIO
import base64
import gradio as gr
import json
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY_FILE = "api_key.json"

SUPPORTED_FORMATS = {'png', 'jpeg', 'gif', 'webp'}
MAX_TOTAL_SIZE_MB = 20
MAX_SIZE_BYTES = MAX_TOTAL_SIZE_MB * 1024 * 1024
PRESETS_FILE = "presets.json"
OUTPUT_PATH_PRESETS_FILE = "output_path_presets.json"

# Token limits for different models
TOKEN_LIMITS = {
    "gpt-3.5-turbo": {"TPM": 60000, "RPM": 500, "RPD": 10000, "TPD": 200000},
    "gpt-4": {"TPM": 10000, "RPM": 500, "RPD": 10000, "TPD": 100000},
    "gpt-4-turbo": {"TPM": 30000, "RPM": 500, "TPD": 90000},
    "gpt-4-vision-preview": {"TPM": 10000, "RPM": 80, "RPD": 500, "TPD": 30000},
    "gpt-4o": {"TPM": 30000, "RPM": 500, "TPD": 90000},
    "gpt-4o-2024-05-13": {"TPM": 30000, "RPM": 500, "TPD": 90000}
}

# Load API key from file if available
def load_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as file:
            data = json.load(file)
            return data.get("api_key", "")
    return ""

# Save API key to file
def save_api_key(api_key):
    with open(API_KEY_FILE, 'w') as file:
        json.dump({"api_key": api_key}, file)

# Load presets from file if available
def load_presets(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

# Save presets to file
def save_presets(presets, file_path):
    with open(file_path, 'w') as file:
        json.dump(presets, file, indent=4)

def is_supported_format(image_format):
    return image_format.lower() in SUPPORTED_FORMATS

def resize_image(image, max_dimension):
    """Resize the image to fit within the max dimension while maintaining aspect ratio."""
    image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    return image

def pad_to_square(image):
    """Pad the image with whitespace to make it a square."""
    max_dimension = max(image.size)
    new_image = Image.new("RGB", (max_dimension, max_dimension), (255, 255, 255))
    new_image.paste(image, ((max_dimension - image.width) // 2, (max_dimension - image.height) // 2))
    return new_image

def encode_image_to_base64(image, image_format):
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def check_total_size(images):
    total_size = sum(len(img) for img in images)
    if total_size > MAX_SIZE_BYTES:
        raise ValueError(f"Total image size exceeds {MAX_TOTAL_SIZE_MB} MB limit.")

def process_images(images, max_image_dimension):
    base64_images = []
    image_filenames = []
    for image_path in images:
        img = Image.open(image_path)

        if not is_supported_format(img.format):
            raise ValueError(f"Unsupported image format: {img.format}. Supported formats are: {SUPPORTED_FORMATS}.")

        img = resize_image(img, max_image_dimension)
        img = pad_to_square(img)
        image_format = img.format if img.format else 'JPEG'
        base64_image = encode_image_to_base64(img, image_format)
        base64_images.append(base64_image)

        # Extract the image filename without extension for saving captions
        filename = os.path.basename(image_path).split('.')[0]
        image_filenames.append(filename)

    check_total_size(base64_images)
    return base64_images, image_filenames

def get_token_limit(model):
    """Retrieve the token limit for the specified model."""
    return TOKEN_LIMITS.get(model, {"TPM": 10000, "RPM": 80, "TPD": 30000})

def generate_image_captions(api_key, model, images, prompt, max_image_dimension, max_tokens=300):
    openai.api_key = api_key
    base64_images, image_filenames = process_images(images, max_image_dimension)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    captions = []

    for img_str in base64_images:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": prompt
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens
        }

        # print(f"Generating caption for image: {img_str}")

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()  # Parse the response JSON

        if 'choices' in response_json:
            captions.append(response_json['choices'][0]['message']['content'].strip())
        else:
            error_message = response_json.get('error', {}).get('message', 'Unknown error')
            print(f"Error in API response: {error_message}")
            captions.append(f"Error: {error_message}")

        print(response_json)
    return captions, image_filenames

def batch_generate_captions(api_key, model, images, prompt, output_path, max_image_dimension, folder_path=None):
    if folder_path:
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.split('.')[-1].lower() in SUPPORTED_FORMATS]

    token_limit = get_token_limit(model)
    max_tokens_per_minute = token_limit["TPM"]
    max_tokens_per_request = min(token_limit["TPM"], 300)  # 300 is a default example limit

    # Split images into smaller batches to stay within token limit
    batch_size = max(1, max_tokens_per_minute // max_tokens_per_request)
    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    all_captions = []

    for batch in batches:
        captions, filenames = generate_image_captions(api_key, model, batch, prompt, max_image_dimension, max_tokens=max_tokens_per_request)
        all_captions.extend(zip(captions, filenames))

    for caption, filename in all_captions:
        save_caption_to_file(caption, filename, output_path)

    return "Captions saved automatically for all images."

def save_caption_to_file(caption, filename, output_path):
    output_file_path = os.path.join(output_path, f"{filename}.txt")
    with open(output_file_path, 'w') as file:
        file.write(caption)
    print(f"Caption saved to: {output_file_path}")

def noise_level(image):
    """Calculate the noise level of the image."""
    gray_image = image.convert("L")
    noise = np.std(np.array(gray_image))
    return noise

def color_distribution(image):
    """Get color distribution (mean and standard deviation of RGB channels)."""
    pixels = np.array(image)
    mean = np.mean(pixels, axis=(0, 1))
    std = np.std(pixels, axis=(0, 1))
    return mean, std

def sharpness(image):
    """Calculate the sharpness of the image."""
    image = image.convert("L")
    image = image.filter(ImageFilter.FIND_EDGES)
    sharpness_value = np.mean(np.array(image))
    return sharpness_value

def frequency_analysis(image):
    """Perform frequency analysis using Fourier Transform."""
    gray_image = image.convert("L")
    f_transform = np.fft.fft2(np.array(gray_image))
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    return np.mean(magnitude_spectrum)

def calculate_image_similarity(image1, image2):
    """Calculate similarity between two images using SSIM."""
    image1 = image1.convert("L")
    image2 = image2.convert("L")
    image1 = np.array(image1)
    image2 = np.array(image2)
    similarity_index, _ = ssim(image1, image2, full=True)
    return similarity_index

def calculate_caption_similarity(captions):
    """Calculate similarity between captions using TF-IDF and Cosine Similarity."""
    vectorizer = TfidfVectorizer().fit_transform(captions)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    return cosine_matrix

def analyze_directory(directory_path):
    image_stats = []
    word_counts = Counter()
    captions = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.split('.')[-1].lower() in SUPPORTED_FORMATS:
                file_path = os.path.join(root, file)
                img = Image.open(file_path)

                # Image statistics
                img_stats = {
                    "Filename": file,
                    "Width": img.width,
                    "Height": img.height,
                    "Format": img.format,
                    "Size (KB)": os.path.getsize(file_path) / 1024,
                    "Noise Level": noise_level(img),
                    "Color Mean": color_distribution(img)[0],
                    "Color Std": color_distribution(img)[1],
                    "Sharpness": sharpness(img),
                    "Frequency Mean": frequency_analysis(img)
                }
                image_stats.append(img_stats)

                caption_file = os.path.join(root, file.split('.')[0] + '.txt')
                if os.path.exists(caption_file):
                    with open(caption_file, 'r') as f:
                        caption = f.read()
                        words = caption.split()
                        word_counts.update(words)
                        captions.append(caption)

    df_images = pd.DataFrame(image_stats)
    df_words = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

    # Image similarity
    if len(image_stats) > 1:
        img_similarities = []
        for i in range(len(image_stats)):
            for j in range(i + 1, len(image_stats)):
                img1_path = os.path.join(directory_path, image_stats[i]["Filename"])
                img2_path = os.path.join(directory_path, image_stats[j]["Filename"])
                img1 = Image.open(img1_path)
                img2 = Image.open(img2_path)
                similarity = calculate_image_similarity(img1, img2)
                img_similarities.append({
                    "Image1": image_stats[i]["Filename"],
                    "Image2": image_stats[j]["Filename"],
                    "Similarity": similarity
                })
        df_img_similarity = pd.DataFrame(img_similarities)
    else:
        df_img_similarity = pd.DataFrame(columns=["Image1", "Image2", "Similarity"])

    # Caption similarity
    if len(captions) > 1:
        caption_sim_matrix = calculate_caption_similarity(captions)
        caption_similarities = []
        for i in range(len(captions)):
            for j in range(i + 1, len(captions)):
                caption_similarities.append({
                    "Caption1": image_stats[i]["Filename"],
                    "Caption2": image_stats[j]["Filename"],
                    "Similarity": caption_sim_matrix[i][j]
                })
        df_caption_similarity = pd.DataFrame(caption_similarities)
    else:
        df_caption_similarity = pd.DataFrame(columns=["Caption1", "Caption2", "Similarity"])

    return df_images, df_words, df_img_similarity, df_caption_similarity

# Load existing presets
presets = load_presets(PRESETS_FILE)
output_path_presets = load_presets(OUTPUT_PATH_PRESETS_FILE)

# Gradio Interface
def single_image_mode(api_key, model, image, prompt, output_path, max_image_dimension):
    captions, filenames = generate_image_captions(api_key, model, [image], prompt, max_image_dimension)
    save_caption_to_file(captions[0], filenames[0], output_path)
    return captions[0]

def batch_image_mode(api_key, model, images, prompt, output_path, max_image_dimension, folder_path=None):
    if folder_path:
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.split('.')[-1].lower() in SUPPORTED_FORMATS]

    token_limit = get_token_limit(model)
    max_tokens_per_minute = token_limit["TPM"]
    max_tokens_per_request = min(token_limit["TPM"], 300)  # 300 is a default example limit

    # Split images into smaller batches to stay within token limit
    batch_size = max(1, max_tokens_per_minute // max_tokens_per_request)
    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    all_captions = []

    for batch in batches:
        captions, filenames = generate_image_captions(api_key, model, batch, prompt, max_image_dimension, max_tokens=max_tokens_per_request)
        all_captions.extend(zip(captions, filenames))

    for caption, filename in all_captions:
        save_caption_to_file(caption, filename, output_path)

    return "Captions saved automatically for all images."

def add_preset(preset_name, prompt):
    presets.append({"name": preset_name, "prompt": prompt})
    save_presets(presets, PRESETS_FILE)
    return gr.update(choices=[preset["name"] for preset in presets])

def delete_preset(preset_name):
    global presets
    presets = [preset for preset in presets if preset["name"] != preset_name]
    save_presets(presets, PRESETS_FILE)
    return gr.update(choices=[preset["name"] for preset in presets])

def load_preset(preset_name):
    for preset in presets:
        if preset["name"] == preset_name:
            return preset["prompt"]
    return ""

def add_output_path_preset(preset_name, output_path):
    output_path_presets.append({"name": preset_name, "output_path": output_path})
    save_presets(output_path_presets, OUTPUT_PATH_PRESETS_FILE)
    return gr.update(choices=[preset["name"] for preset in output_path_presets])

def delete_output_path_preset(preset_name):
    global output_path_presets
    output_path_presets = [preset for preset in output_path_presets if preset["name"] != preset_name]
    save_presets(output_path_presets, OUTPUT_PATH_PRESETS_FILE)
    return gr.update(choices=[preset["name"] for preset in output_path_presets])

def load_output_path_preset(preset_name):
    for preset in output_path_presets:
        if preset["name"] == preset_name:
            return preset["output_path"]
    return ""

def analyze_stats(directory_path):
    df_images, df_words, df_img_similarity, df_caption_similarity = analyze_directory(directory_path)

    # Image stats plot
    fig, ax = plt.subplots()
    df_images[['Width', 'Height']].plot(kind='hist', bins=30, alpha=0.5, ax=ax)
    ax.set_title("Image Dimension Distribution")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("image_stats.png")

    # Word frequency plot
    fig, ax = plt.subplots()
    df_words.head(20).plot(kind='bar', x='Word', y='Frequency', ax=ax)
    ax.set_title("Top 20 Words Frequency")
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("word_stats.png")

    return df_images, df_words, df_img_similarity, df_caption_similarity

with gr.Blocks() as demo:
    gr.Markdown("## Image Caption Generator with GPT-4")

    api_key_input = gr.Textbox(label="API Key", value=load_api_key(), type="password")
    save_api_key_button = gr.Button("Save API Key", variant="primary")
    model_selection = gr.Dropdown(label="Model", choices=["gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], value="gpt-4o")
    prompt_input = gr.Textbox(label="Prompt", lines=2)
    output_path_input = gr.Textbox(label="Output Path")
    max_image_dimension_slider = gr.Slider(label="Max Image Dimension", minimum=128, maximum=1024, step=64, value=512)

    output_path_preset_name_input = gr.Textbox(label="Output Path Preset Name")
    output_path_presets_dropdown = gr.Dropdown(label="Output Path Presets", choices=[preset["name"] for preset in output_path_presets])
    with gr.Row():
        load_output_path_preset_button = gr.Button("Load Output Path Preset", variant="primary")
        delete_output_path_preset_button = gr.Button("Delete Output Path Preset", variant="primary")
        add_output_path_preset_button = gr.Button("Add Output Path Preset", variant="primary")

    with gr.Tabs():
        with gr.TabItem("Single Image Mode"):
            image_input = gr.Image(type="filepath", label="Upload Image")
            single_image_output = gr.Textbox(label="Generated Caption")
            with gr.Row():
                save_single_image_button = gr.Button("Save Caption", variant="primary")
                single_image_button = gr.Button("Generate Caption", variant="primary")
            single_image_button.click(single_image_mode, [api_key_input, model_selection, image_input, prompt_input, output_path_input, max_image_dimension_slider], single_image_output)
            save_single_image_button.click(save_caption_to_file, [single_image_output, image_input, output_path_input], outputs=[])

        with gr.TabItem("Batch Image Mode"):
            images_input = gr.Files(type="filepath", label="Upload Images", file_count="multiple")
            folder_path_input = gr.Textbox(label="Folder Path (optional)")
            batch_image_output = gr.Textbox(label="Status")
            batch_image_button = gr.Button("Generate Captions", variant="primary")
            batch_image_button.click(batch_image_mode, [api_key_input, model_selection, images_input, prompt_input, output_path_input, max_image_dimension_slider, folder_path_input], batch_image_output)

        with gr.TabItem("Statistics"):
            directory_input = gr.Textbox(label="Directory Path")
            stats_button = gr.Button("Generate Stats", variant="primary")
            image_stats_output = gr.Dataframe(label="Image Stats")
            word_stats_output = gr.Dataframe(label="Word Stats")
            img_similarity_output = gr.Dataframe(label="Image Similarity")
            caption_similarity_output = gr.Dataframe(label="Caption Similarity")
            stats_button.click(analyze_stats, directory_input, [image_stats_output, word_stats_output, img_similarity_output, caption_similarity_output])

    with gr.Accordion("Presets", open=True):
        with gr.Column():
            preset_name_input = gr.Textbox(label="Preset Name")
            presets_dropdown = gr.Dropdown(label="Presets", choices=[preset["name"] for preset in presets])
            with gr.Row():
                load_preset_button = gr.Button("Load Preset", variant="primary")
                delete_preset_button = gr.Button("Delete Preset", variant="primary")
                add_preset_button = gr.Button("Add Preset", variant="primary")

        add_preset_button.click(add_preset, [preset_name_input, prompt_input], presets_dropdown)
        delete_preset_button.click(delete_preset, presets_dropdown, presets_dropdown)
        load_preset_button.click(load_preset, presets_dropdown, prompt_input)

    save_api_key_button.click(save_api_key, inputs=[api_key_input], outputs=[])
    add_output_path_preset_button.click(add_output_path_preset, [output_path_preset_name_input, output_path_input], output_path_presets_dropdown)
    delete_output_path_preset_button.click(delete_output_path_preset, output_path_presets_dropdown, output_path_presets_dropdown)
    load_output_path_preset_button.click(load_output_path_preset, output_path_presets_dropdown, output_path_input)

demo.launch()
