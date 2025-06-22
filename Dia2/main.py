from src.trainer import WalmartPulseTrainer
from src.config import MODEL_CONFIG

def main():
    print("Walmart Pulse Demand Forecasting System")
    print("1. Train Model")
    print("2. Make Predictions")
    
    choice = input("Select option (1/2): ")
    
    if choice == "1":
        print("\nTraining model...")
        trainer = WalmartPulseTrainer()
        trainer.train()
        print("Model training complete! Saved to models/ folder")
    
    elif choice == "2":
        print("\nLoading model for predictions...")
        # Example usage - in practice you'd load your data
        predictor = WalmartPulsePredictor(
            input_shape=(MODEL_CONFIG['sequence_length'], 7)  # 7 features
        )
        print("Predictor ready! Use predictor.predict() with your data")
    
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()