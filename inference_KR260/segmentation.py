import cv2
import numpy as np
import xir
import vart
import time

def main():
    #model_path = "semantic_ulite_custom5_cont_kr260.xmodel"
    model_path = "semantic_ulite_custom5_cont_kr260.xmodel" # Your newly compiled model
    image_path = "test_image_11.jpg"

    print("[INFO] Loading XIR Graph...")
    graph = xir.Graph.deserialize(model_path)
    root_subgraph = graph.get_root_subgraph()
    subgraphs = root_subgraph.toposort_child_subgraph()
    
    dpu_subgraphs = [s for s in subgraphs if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    if len(dpu_subgraphs) == 0:
        raise ValueError("No DPU subgraph found in the model.")
    
    print("[INFO] Initializing VART Runner...")
    runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")

    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    
    in_dims = input_tensors[0].dims
    out_dims = output_tensors[0].dims
    batch, in_height, in_width, in_channels = in_dims
    _, out_height, out_width, padded_classes = out_dims

    print(f"[INFO] Expected Input Shape: {in_width}x{in_height}")

    # 4. Preprocess the Image (UPDATED TO MATCH YOUR PYTORCH TRAINING)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (in_width, in_height))

    # PyTorch transforms.ToTensor() scales pixels to 0.0 - 1.0
    img_float = img_resized.astype(np.float32) / 255.0
    #img_float = img_resized.astype(np.float32)

    # Your custom drone dataset Mean and Std from test.py
    mean = np.array([0.31328324, 0.32151696, 0.31460182], dtype=np.float32)
    std = np.array([0.23343998, 0.24014007, 0.23295579], dtype=np.float32)

    # PyTorch Normalize does (image - mean) / std
    img_normalized = (img_float - mean) / std

    # Apply the DPU's internal Quantization Factor
    fix_point = input_tensors[0].get_attr("fix_point")
    quant_scale = 2 ** fix_point
    img_quantized = img_normalized * quant_scale
    img_int8 = np.asarray(img_quantized, dtype=np.int8)

    # Prepare arrays
    input_data = [np.empty(in_dims, dtype=np.int8)]
    output_data = [np.empty(out_dims, dtype=np.int8)]
    input_data[0][0] = img_int8

    print("[INFO] Running Hardware-Accelerated Inference...")
    start_time = time.perf_counter()
    job_id = runner.execute_async(input_data, output_data)
    runner.wait(job_id)
    
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    fps = 1.0 / inference_time

    print("[INFO] Post-processing Output Mask...")
    # 6. Post-Process the Output (UPDATED TO FIX DPU PADDING)
    out_fix_point = output_tensors[0].get_attr("fix_point")
    out_scale = 2 ** out_fix_point
    
    # Slice only the 0th channel (your actual prediction) and ignore the 15 hardware-padded channels
    raw_output = output_data[0][0][:, :, 0] 
    valid_logits_int8 = output_data[0][0][:, :, :2]
    valid_logits_int8 = valid_logits_int8[:,:, ::-1] 
    print(valid_logits_int8.shape)
    
    # Dequantize back to floating point logits
    # logits = raw_output / out_scale
    logits_float = valid_logits_int8 / out_scale
    
    # Threshold at 0.0 for binary segmentation (drone vs background)
    # segmentation_mask = (logits > 0.0).astype(np.uint8)
    segmentation_mask = np.argmax(logits_float, axis=-1).astype(np.uint8)

    # 7. Visualize the Result (UPDATED FOR 1 CLASS)
    # Mask is 0 or 1. Multiply by 255 to make it pure white
    mask_visual = segmentation_mask * 255
    
    # Convert to 3-channel BGR so we can colorize and blend it
    color_mask = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)
    
    # Tint the mask Green for better visibility over the drone
    color_mask[:, :, 0] = 0   # Set Blue to 0
    color_mask[:, :, 2] = 0   # Set Red to 0

    # Resize mask back to original image size
    color_mask_resized = cv2.resize(color_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Blend original image and the green mask
    blended_img = cv2.addWeighted(img, 0.7, color_mask_resized, 0.5, 0)

    # Save output
    cv2.imwrite("mask.jpg", color_mask_resized)
    cv2.imwrite("segmentation_result.jpg", blended_img)
    print("[INFO] Success! Saved to segmentation_result.jpg")
    print(f"[INFO] DPU Inference Time: {inference_time:.4f} seconds")
    print(f"[INFO] Estimated Speed: {fps:.2f} FPS")

if __name__ == "__main__":
    main()
