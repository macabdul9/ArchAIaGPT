#!/usr/bin/env python3
import argparse
import numpy as np
import sys
from pathlib import Path
from PIL import Image

# Ensure the project root is in the search path so we can import internal modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from embeddings.factory import get_encoder

def main():
    """
    Entry point for calculating embeddings for a single text string or image file.
    This utility is useful for research and manual verification of model outputs.
    """
    parser = argparse.ArgumentParser(description="Calculate embeddings for a single image and text query.")
    parser.add_argument("--text", type=str, default=None, help="Text to embed")
    parser.add_argument("--image_path", type=str, default=None, help="Path to image to embed")
    parser.add_argument("--text_model", type=str, default="gemma", help="Model type for text (clip, gemma, etc.)")
    parser.add_argument("--image_model", type=str, default="clip", help="Model type for image (clip, vlm2vec, etc.)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save .npy files")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    if not args.text and not args.image_path:
        print("Error: Must provide either --text or --image_path")
        sys.exit(1)
        
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process text embedding if a query was provided
    if args.text:
        print(f"Loading text encoder: {args.text_model}")
        text_encoder = get_encoder(args.text_model, device=args.device)
        print(f"Encoding text: '{args.text}'")
        text_embedding = text_encoder.encode_texts([args.text])[0]
        
        output_path = out_dir / "text.npy"
        np.save(output_path, text_embedding)
        print(f"Saved text embedding to {output_path} (shape: {text_embedding.shape})")
        
    # Process image embedding if a path was provided
    if args.image_path:
        img_path = Path(args.image_path)
        if not img_path.exists():
            print(f"Error: Image file not found at {args.image_path}")
        else:
            print(f"Loading image encoder: {args.image_model}")
            image_encoder = get_encoder(args.image_model, device=args.device)
            print(f"Encoding image: {args.image_path}")
            
            # Open and preprocess the image
            image = Image.open(img_path).convert("RGB")
            
            # Check if the selected model supports image encoding
            if hasattr(image_encoder, "encode_images"):
                image_embedding = image_encoder.encode_images([image])[0]
                
                # Verify that the encoder returned a valid vector
                if image_embedding is not None:
                    output_path = out_dir / "image.npy"
                    np.save(output_path, image_embedding)
                    print(f"Saved image embedding to {output_path} (shape: {image_embedding.shape})")
                else:
                    print(f"Warning: Encoder {args.image_model} returned None for the provided image.")
            else:
                print(f"Warning: Selected model {args.image_model} does not support image features.")

if __name__ == "__main__":
    main()
