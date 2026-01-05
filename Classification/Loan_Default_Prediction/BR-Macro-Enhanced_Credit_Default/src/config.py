import os

class Paths:
    """
    Manages all file and directory paths for the project.
    
    Attributes:
        PROJECT_BASE_PATH (str): The root directory of the project.
    """
    
    def __init__(self, project_base_path):
        """
        Initializes the Paths class and defines core directories.
        
        Args:
            project_base_path (str): The absolute path to the 'Project_01' folder.
        """
        self.PROJECT_BASE_PATH = project_base_path
        
        # --- Core Directories ---
        self.DATA_DIR = os.path.join(self.PROJECT_BASE_PATH, 'data')
        self.DATA_RAW_DIR = os.path.join(self.DATA_DIR, 'raw')
        self.DATA_PROCESSED_DIR = os.path.join(self.DATA_DIR, 'processed')
        self.MODEL_DIR = os.path.join(self.PROJECT_BASE_PATH, 'models') # For saving trained models (e.g., model.pkl)
        self.REPORT_DIR = os.path.join(self.PROJECT_BASE_PATH, 'reports') # For EDA plots, performance metrics, etc.
        self.SUBMISSION_DIR = os.path.join(self.PROJECT_BASE_PATH, 'submissions') # For predicted results
        self.SRC_DIR = os.path.join(self.PROJECT_BASE_PATH, 'src') # source python scripts

        # --- File Paths ---
        
        # Raw Data (Input)
        self.TRAIN_RAW_FILE = os.path.join(self.DATA_RAW_DIR, 'application_train.csv')
        self.TEST_RAW_FILE = os.path.join(self.DATA_RAW_DIR, 'application_test.csv')
        self.MACRO_RAW_FILE = os.path.join(self.DATA_RAW_DIR, 'brasil_macro_data.csv') # Assuming you save the BCB data here

        # Processed Data (Output of Block 9)
        self.TRAIN_PROCESSED_FILE = os.path.join(self.DATA_PROCESSED_DIR, 'train_enriched.csv')
        self.TEST_PROCESSED_FILE = os.path.join(self.DATA_PROCESSED_DIR, 'test_enriched.csv')


    def create_dirs(self):
        """Creates all necessary directories if they don't exist."""
        for path in [self.DATA_RAW_DIR, self.DATA_PROCESSED_DIR, self.MODEL_DIR, self.REPORT_DIR, self.SUBMISSION_DIR, self.SRC_DIR] :
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created directory: {path}")