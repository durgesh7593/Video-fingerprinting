# processing/your_processing_script.py

import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from algorithms import (
    perceptual_hash,
    difference_hash,
    average_hash,
    block_mean_value_hash,
    sift,
    fmt,
    color_histogram,
    wavelet,
    # cnn_fingerprinting,
    motion_vector,
    spatio_temporal,
    keyframe_extraction,
    frame_signature
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def write_to_file(filename, content):
    """Write content to a specified file."""
    try:
        with open(filename, 'w') as file:
            file.write(content)
        logging.info(f"Successfully wrote to {filename}")
    except Exception as e:
        logging.error(f"Failed to write to {filename}: {e}")

def format_results(algorithm_name, result):
    """Format the results into a readable string."""
    formatted_str = f"Algorithm: {algorithm_name}\n"
    formatted_str += f"Total Frames Processed: {result.get('total_frames_processed', 'N/A')}\n"
    formatted_str += f"Total Frames in Video: {result.get('total_frames_in_video', 'N/A')}\n"
    formatted_str += f"Time Required to Generate Fingerprints: {result.get('execution_time', 'N/A')} seconds\n"
    formatted_str += f"Time Complexity: {result.get('time_complexity', 'N/A')}\n"
    formatted_str += f"Space Complexity: {result.get('space_complexity', 'N/A')}\n"
    # Add additional fields if necessary
    return formatted_str

def main(video_path1, video_path2, output_directory):
    """
    Main processing function that orchestrates the execution of various algorithms
    on the provided video files and generates output reports and visualizations.
    
    Args:
        video_path1 (str): Path to the first video file.
        video_path2 (str): Path to the second video file.
        output_directory (str): Directory where output files will be saved.
    """
    
    logging.info("Starting video processing...")
    
    # Create directory for outputs if it doesn't exist
    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
            logging.info(f"Created output directory at {output_directory}")
        except Exception as e:
            logging.error(f"Failed to create output directory: {e}")
            return
    
    # List of algorithms to execute
    algorithmnames = [
        (perceptual_hash, "perceptual_hash"),
        (difference_hash, "difference_hash"),
        (average_hash, "average_hash"),
        (block_mean_value_hash, "block_mean_value_hash"),
        (sift, "sift"),
        (fmt, "fmt"),
        (color_histogram, "color_histogram"),
        # (cnn_fingerprinting,"cnn_algorithm"),
        # (wavelet, "wavelet"),
        (motion_vector, "motion_vector"),
        (spatio_temporal, "spatio_temporal"),
        (keyframe_extraction, "keyframe_extraction"),
        (frame_signature, "frame_signature")
        # Add other algorithms here if necessary
    ]
    
    # Initialize lists to store comparison data
    comparison_data = []
    execution_times = []
    
    # Run each algorithm and write the result to a file
    for algorithm, name in algorithmnames:
        logging.info(f"Processing algorithm: {name}")
        
        # Compute fingerprints
        try:
            start_time = time.time()
            result = algorithm.compute_video_fingerprints(video_path1, video_path2)
            end_time = time.time()
            result['execution_time'] = end_time - start_time
            logging.info(f"Completed fingerprint generation for {name} in {result['execution_time']:.2f} seconds")
        except Exception as e:
            logging.error(f"Error processing {name}: {e}")
            continue
        
        formatted_result = format_results(name, result)
        
        # Collect execution time for fingerprint generation histogram
        execution_times.append((name, result.get('execution_time', 0)))
        
        # Check determinism or other checks if needed
        try:
            start_determinism = time.time()
            determinism_result = algorithm.check_determinism(video_path1)
            end_determinism = time.time()
            determinism_time = end_determinism - start_determinism
            formatted_result += f"Determinism Check: {determinism_result}\n"
            logging.info(f"Determinism check for {name}: {determinism_result} (Time: {determinism_time:.2f} seconds)")
        except AttributeError:
            # If the algorithm doesn't have a check_determinism method
            determinism_result = "N/A"
            formatted_result += f"Determinism Check: {determinism_result}\n"
            logging.warning(f"Determinism check not available for {name}")
        except Exception as e:
            determinism_result = f"Error: {e}"
            formatted_result += f"Determinism Check: {determinism_result}\n"
            logging.error(f"Error during determinism check for {name}: {e}")
        
        # Append data for comparison table
        comparison_data.append({
            'Algorithm': name,
            'Total Frames Processed': result.get('total_frames_processed', 'N/A'),
            'Total Frames in Video': result.get('total_frames_in_video', 'N/A'),
            'Execution Time (s)': f"{result.get('execution_time', 0):.2f}",
            'Time Complexity': result.get('time_complexity', 'N/A'),
            'Space Complexity': result.get('space_complexity', 'N/A'),
            'Determinism Check': determinism_result
            # Add more fields if necessary
        })
        
        # Write formatted results to file
        filename = os.path.join(output_directory, f"{name}_results.txt")
        write_to_file(filename, formatted_result)
        logging.info(f"Results for {name} saved to {filename}\n")
    
    # Plot histogram of execution times
    if execution_times:
        algorithm_names, times = zip(*execution_times)
        plt.figure(figsize=(12, 8))
        bars = plt.bar(algorithm_names, times, color='skyblue')
        plt.xlabel('Algorithms')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Histogram of Execution Times for Fingerprint Generation')
        plt.xticks(rotation=45, ha='right')
        
        # Annotate bars with execution time
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{time_val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        histogram_path = os.path.join(output_directory, "execution_times_histogram.png")
        try:
            plt.savefig(histogram_path)
            plt.close()
            logging.info(f"Execution times histogram saved to {histogram_path}\n")
        except Exception as e:
            logging.error(f"Failed to save execution times histogram: {e}")
    else:
        logging.warning("No execution times to plot.\n")
    
    # Similarity checks between video1 and video2
    similarity_results = []
    similarity_times = []
    
    for algorithm, name in algorithmnames:
        logging.info(f"Checking similarity using algorithm: {name}")
        try:
            start_time = time.time()
            is_same = algorithm.isSame(video_path1, video_path2)
            end_time = time.time()
            similarity_time = end_time - start_time
            logging.info(f"Similarity check by {name}: {'Same' if is_same else 'Different'} (Time: {similarity_time:.2f} seconds)")
        except AttributeError:
            # If the algorithm doesn't have an isSame method
            is_same = "N/A"
            similarity_time = "N/A"
            logging.warning(f"isSame method not available for {name}")
        except Exception as e:
            is_same = f"Error: {e}"
            similarity_time = "Error"
            logging.error(f"Error checking similarity with {name}: {e}")
        
        similarity_results.append({
            'Algorithm': name,
            'v1_vs_v2_Same': is_same,
            'Similarity Check Time (s)': similarity_time
        })
        similarity_times.append((name, similarity_time))
        
        if isinstance(is_same, bool):
            logging.info(f"Similarity check by {name}: {'Same' if is_same else 'Different'} (Time: {similarity_time:.2f} seconds)\n")
        elif similarity_time == "N/A":
            logging.info(f"Similarity check by {name}: {is_same}\n")
        elif similarity_time == "Error":
            logging.info(f"Similarity check by {name}: {is_same}\n")
        else:
            logging.info(f"Similarity check by {name}: {is_same} (Time: {similarity_time:.2f} seconds)\n")
    
    # Merge comparison data with similarity results
    comparison_df = pd.DataFrame(comparison_data)
    similarity_df = pd.DataFrame(similarity_results)
    full_comparison_df = pd.merge(comparison_df, similarity_df, on='Algorithm')
    
    # Save comparison table to a CSV file
    comparison_filename = os.path.join(output_directory, "comparison_table.csv")
    try:
        full_comparison_df.to_csv(comparison_filename, index=False)
        logging.info(f"Comparison table saved to {comparison_filename}\n")
    except Exception as e:
        logging.error(f"Failed to save comparison table CSV: {e}")
    
    # Save the table as a text file
    table_str = full_comparison_df.to_string(index=False)
    table_txt_filename = os.path.join(output_directory, "comparison_table.txt")
    write_to_file(table_txt_filename, table_str)
    logging.info(f"Comparison table (text) saved to {table_txt_filename}\n")
    
    # Save the table as a Markdown file
    try:
        table_md = full_comparison_df.to_markdown(index=False)
        table_md_filename = os.path.join(output_directory, "comparison_table.md")
        write_to_file(table_md_filename, table_md)
        logging.info(f"Comparison table (Markdown) saved to {table_md_filename}\n")
    except Exception as e:
        logging.error(f"Failed to save comparison table Markdown: {e}")
    
    # Plot histogram for similarity check times
    valid_similarity_times = [
        (name, time_val) for name, time_val in similarity_times
        if isinstance(time_val, (int, float))
    ]
    if valid_similarity_times:
        sim_names, sim_times = zip(*valid_similarity_times)
        plt.figure(figsize=(12, 8))
        sim_bars = plt.bar(sim_names, sim_times, color='salmon')
        plt.xlabel('Algorithms')
        plt.ylabel('Similarity Check Time (seconds)')
        plt.title('Histogram of Similarity Check Times')
        plt.xticks(rotation=45, ha='right')
        
        # Annotate bars with similarity check time
        for bar, sim_time in zip(sim_bars, sim_times):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{sim_time:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        sim_histogram_path = os.path.join(output_directory, "similarity_check_times_histogram.png")
        try:
            plt.savefig(sim_histogram_path)
            plt.close()
            logging.info(f"Similarity check times histogram saved to {sim_histogram_path}\n")
        except Exception as e:
            logging.error(f"Failed to save similarity check times histogram: {e}")
    else:
        logging.warning("No valid similarity check times to plot.\n")
    
    logging.info("Video processing completed successfully.")

