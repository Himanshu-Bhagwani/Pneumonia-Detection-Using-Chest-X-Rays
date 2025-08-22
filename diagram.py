# minor/generate_mermaid_diagram.py

# This script generates diagram syntax for Mermaid (https://mermaid.live/)
# VERSION WITH DARKER TEXT/BORDERS IN STYLES

def generate_mermaid_syntax():
    """Generates the Mermaid syntax for the project workflow with darker styles."""

    mermaid_code = ["graph TD"]
    indent = "    "

    # --- Define Nodes (Using simplified labels from before) ---
    mermaid_code.append(f'{indent}subgraph "1_Data"')
    mermaid_code.append(f'{indent}{indent}N_KaggleData["Kaggle_Dataset"]:::data')
    mermaid_code.append(f'{indent}{indent}N_TrainData["Full_Training_Set"]:::data')
    mermaid_code.append(f'{indent}{indent}N_ValData["Validation_Split"]:::data')
    mermaid_code.append(f'{indent}{indent}N_TestData["Test_Set"]:::data')
    mermaid_code.append(f'{indent}end')

    mermaid_code.append(f'{indent}subgraph "2a_LeViT_Path"')
    mermaid_code.append(f'{indent}{indent}N_LevitModelDef["Levit_Model_Py"]:::script')
    mermaid_code.append(f'{indent}{indent}N_TrainLevit["Train_Levit_Py"]:::script')
    mermaid_code.append(f'{indent}{indent}N_LevitWeights["Levit_Weights"]:::model')
    mermaid_code.append(f'{indent}{indent}N_EvalLevit["Evaluate_Levit_Py"]:::script')
    mermaid_code.append(f'{indent}{indent}N_LevitMetrics["Levit_Metrics"]:::output')
    mermaid_code.append(f'{indent}end')

    mermaid_code.append(f'{indent}subgraph "2b_ResNet50_Benchmark"')
    mermaid_code.append(f'{indent}{indent}N_ResnetModelDef["Torchvision_ResNet50"]:::script')
    mermaid_code.append(f'{indent}{indent}N_TrainResnet["Train_Resnet_Py"]:::script')
    mermaid_code.append(f'{indent}{indent}N_ResnetWeights["Resnet50_Weights"]:::model')
    mermaid_code.append(f'{indent}{indent}N_EvalResnet["Evaluate_Resnet_Py"]:::script')
    mermaid_code.append(f'{indent}{indent}N_ResnetMetrics["Resnet50_Metrics"]:::output')
    mermaid_code.append(f'{indent}end')

    mermaid_code.append(f'{indent}subgraph "3_Comparison"')
    mermaid_code.append(f'{indent}{indent}N_CompareScript["Compare_Models_Py"]:::script')
    mermaid_code.append(f'{indent}{indent}N_CompareResults["Comparison_Report"]:::output')
    mermaid_code.append(f'{indent}end')

    mermaid_code.append(f'{indent}subgraph "4_Application"')
    mermaid_code.append(f'{indent}{indent}N_InferenceScript["Inference_Py"]:::script')
    mermaid_code.append(f'{indent}{indent}N_StreamlitApp["App_Py_UI"]:::app')
    mermaid_code.append(f'{indent}{indent}N_UserImage["User_Image"]:::data')
    mermaid_code.append(f'{indent}{indent}N_AppOutput["Prediction_Output"]:::output')
    mermaid_code.append(f'{indent}end')

    # --- Define Edges (Connections - remain the same) ---
    mermaid_code.append("\n    %% Data Flow")
    mermaid_code.append(f'{indent}N_KaggleData --> N_TrainData')
    mermaid_code.append(f'{indent}N_KaggleData -- Separate --> N_TestData')
    mermaid_code.append(f'{indent}N_TrainData -- Split --> N_ValData')

    mermaid_code.append("\n    %% LeViT Training Flow")
    mermaid_code.append(f'{indent}N_LevitModelDef --> N_TrainLevit')
    mermaid_code.append(f'{indent}N_TrainData --> N_TrainLevit')
    mermaid_code.append(f'{indent}N_ValData --> N_TrainLevit')
    mermaid_code.append(f'{indent}N_TrainLevit --> N_LevitWeights')

    mermaid_code.append("\n    %% LeViT Evaluation Flow")
    mermaid_code.append(f'{indent}N_LevitWeights --> N_EvalLevit')
    mermaid_code.append(f'{indent}N_TestData --> N_EvalLevit')
    mermaid_code.append(f'{indent}N_EvalLevit --> N_LevitMetrics')

    mermaid_code.append("\n    %% ResNet Training Flow")
    mermaid_code.append(f'{indent}N_ResnetModelDef --> N_TrainResnet')
    mermaid_code.append(f'{indent}N_TrainData --> N_TrainResnet')
    mermaid_code.append(f'{indent}N_ValData --> N_TrainResnet')
    mermaid_code.append(f'{indent}N_TrainResnet --> N_ResnetWeights')

    mermaid_code.append("\n    %% ResNet Evaluation Flow")
    mermaid_code.append(f'{indent}N_ResnetWeights --> N_EvalResnet')
    mermaid_code.append(f'{indent}N_TestData --> N_EvalResnet')
    mermaid_code.append(f'{indent}N_EvalResnet --> N_ResnetMetrics')

    mermaid_code.append("\n    %% Comparison Flow")
    mermaid_code.append(f'{indent}N_LevitMetrics --> N_CompareScript')
    mermaid_code.append(f'{indent}N_ResnetMetrics --> N_CompareScript')
    mermaid_code.append(f'{indent}N_CompareScript --> N_CompareResults')

    mermaid_code.append("\n    %% Application Flow")
    mermaid_code.append(f'{indent}N_LevitModelDef --> N_InferenceScript')
    mermaid_code.append(f'{indent}N_LevitWeights --> N_InferenceScript')
    mermaid_code.append(f'{indent}N_UserImage --> N_StreamlitApp')
    mermaid_code.append(f'{indent}N_InferenceScript --> N_StreamlitApp')
    mermaid_code.append(f'{indent}N_StreamlitApp --> N_AppOutput')

    # --- Define Styles with Darker Text and Borders ---
    # Added color:#333 (dark gray text) and darker stroke colors
    mermaid_code.append("\n    %% Styles")
    mermaid_code.append(f'{indent}classDef data fill:#e6f7ff,stroke:#59a9d4,stroke-width:2px,color:#333')
    mermaid_code.append(f'{indent}classDef script fill:#f0f0f0,stroke:#888,stroke-width:2px,color:#333')
    mermaid_code.append(f'{indent}classDef model fill:#fffbe6,stroke:#d4b106,stroke-width:2px,color:#333')
    mermaid_code.append(f'{indent}classDef output fill:#f6ffed,stroke:#7cb305,stroke-width:2px,color:#333')
    mermaid_code.append(f'{indent}classDef app fill:#fff0f6,stroke:#d438a4,stroke-width:2px,color:#333')

    return "\n".join(mermaid_code)

# --- Main Execution ---
if __name__ == "__main__":
    mermaid_output = generate_mermaid_syntax()
    print("--- Mermaid Diagram Syntax (Darker Styles) ---")
    print(mermaid_output)
    print("\n--- How to View ---")
    print("1. Copy the text above (starting with 'graph TD').")
    print("2. Paste it into a Mermaid renderer (e.g., https://mermaid.live/).")