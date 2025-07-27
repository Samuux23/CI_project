"""
Simple runner script for the step-by-step ML pipeline
"""

# For Google Colab: Upload your dataset
try:
    from google.colab import files
    print("Running in Google Colab - please upload your dataset files")
    uploaded = files.upload()
    print("Files uploaded successfully!")
except ImportError:
    print("Running locally - make sure train.csv and test.csv are in the current directory")
    # Create dummy uploaded dict for local execution
    uploaded = {}

# Run the main pipeline
if __name__ == "__main__":
    print("Starting the step-by-step ML pipeline...")
    exec(open('step_by_step_ml_pipeline.py').read()) 