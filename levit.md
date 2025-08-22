```mermaid
graph TD
    subgraph "1_Data"
        N_KaggleData["Kaggle_Dataset"]:::data
        N_TrainData["Full_Training_Set"]:::data
        N_ValData["Validation_Split"]:::data
        N_TestData["Test_Set"]:::data
    end
    subgraph "2a_LeViT_Path"
        N_LevitModelDef["Levit_Model_Py"]:::script
        N_TrainLevit["Train_Levit_Py"]:::script
        N_LevitWeights["Levit_Weights"]:::model
        N_EvalLevit["Evaluate_Levit_Py"]:::script
        N_LevitMetrics["Levit_Metrics"]:::output
    end
    subgraph "2b_ResNet50_Benchmark"
        N_ResnetModelDef["Torchvision_ResNet50"]:::script
        N_TrainResnet["Train_Resnet_Py"]:::script
        N_ResnetWeights["Resnet50_Weights"]:::model
        N_EvalResnet["Evaluate_Resnet_Py"]:::script
        N_ResnetMetrics["Resnet50_Metrics"]:::output
    end
    subgraph "3_Comparison"
        N_CompareScript["Compare_Models_Py"]:::script
        N_CompareResults["Comparison_Report"]:::output
    end
    subgraph "4_Application"
        N_InferenceScript["Inference_Py"]:::script
        N_StreamlitApp["App_Py_UI"]:::app
        N_UserImage["User_Image"]:::data
        N_AppOutput["Prediction_Output"]:::output
    end

    %% Data Flow
    N_KaggleData --> N_TrainData
    N_KaggleData -- Separate --> N_TestData
    N_TrainData -- Split --> N_ValData

    %% LeViT Training Flow
    N_LevitModelDef --> N_TrainLevit
    N_TrainData --> N_TrainLevit
    N_ValData --> N_TrainLevit
    N_TrainLevit --> N_LevitWeights

    %% LeViT Evaluation Flow
    N_LevitWeights --> N_EvalLevit
    N_TestData --> N_EvalLevit
    N_EvalLevit --> N_LevitMetrics

    %% ResNet Training Flow
    N_ResnetModelDef --> N_TrainResnet
    N_TrainData --> N_TrainResnet
    N_ValData --> N_TrainResnet
    N_TrainResnet --> N_ResnetWeights

    %% ResNet Evaluation Flow
    N_ResnetWeights --> N_EvalResnet
    N_TestData --> N_EvalResnet
    N_EvalResnet --> N_ResnetMetrics

    %% Comparison Flow
    N_LevitMetrics --> N_CompareScript
    N_ResnetMetrics --> N_CompareScript
    N_CompareScript --> N_CompareResults

    %% Application Flow
    N_LevitModelDef --> N_InferenceScript
    N_LevitWeights --> N_InferenceScript
    N_UserImage --> N_StreamlitApp
    N_InferenceScript --> N_StreamlitApp
    N_StreamlitApp --> N_AppOutput

    %% Styles
    classDef data fill:#e6f7ff,stroke:#59a9d4,stroke-width:2px,color:#333
    classDef script fill:#f0f0f0,stroke:#888,stroke-width:2px,color:#333
    classDef model fill:#fffbe6,stroke:#d4b106,stroke-width:2px,color:#333
    classDef output fill:#f6ffed,stroke:#7cb305,stroke-width:2px,color:#333
    classDef app fill:#fff0f6,stroke:#d438a4,stroke-width:2px,color:#333