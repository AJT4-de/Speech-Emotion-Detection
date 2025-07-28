#!/usr/bin/env python3
"""
Advanced Emotion Recognition Training Pipeline
Integrates advanced feature extraction with ensemble models and hyperparameter optimization
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Import our modules
from optimized_config import Config
from optimized_data_loader import DataLoader
from advanced_feature_extractor import AdvancedFeatureExtractor
from advanced_model_trainer import AdvancedModelTrainer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('advanced_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main training pipeline with advanced features and models"""
    logger = setup_logging()
    logger.info("Starting Advanced Emotion Recognition Training Pipeline")
    
    start_time = time.time()
    
    try:
        # Initialize components
        config = Config()
        data_loader = DataLoader(config)
        feature_extractor = AdvancedFeatureExtractor(config)
        model_trainer = AdvancedModelTrainer(config)
        
        # Create models directory
        os.makedirs(config.models_dir, exist_ok=True)
        
        # Step 1: Load and validate data
        logger.info("Step 1: Loading and validating dataset...")
        file_paths, labels = data_loader.load_data()
        logger.info(f"Loaded {len(file_paths)} audio files")
        
        # Display class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        logger.info("Class distribution:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"  {label}: {count} files")
        
        # Step 2: Extract features
        logger.info("Step 2: Extracting advanced features...")
        features = feature_extractor.extract_features_batch(file_paths, labels)
        
        if features is None or len(features) == 0:
            logger.error("Feature extraction failed!")
            return False
        
        logger.info(f"Extracted features shape: {features.shape}")
        logger.info(f"Feature dimensionality: {features.shape[1]} features per sample")
        
        # Step 3: Train advanced models
        logger.info("Step 3: Training advanced ensemble models...")
        success = model_trainer.train(features, labels)
        
        if not success:
            logger.error("Model training failed!")
            return False
        
        # Step 4: Evaluate models
        logger.info("Step 4: Evaluating trained models...")
        evaluation_results = model_trainer.evaluate_models(features, labels)
        
        if evaluation_results:
            logger.info("Model Evaluation Results:")
            for model_name, metrics in evaluation_results.items():
                logger.info(f"\n{model_name}:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall: {metrics['recall']:.4f}")
                logger.info(f"  F1-Score: {metrics['f1']:.4f}")
        
        # Step 5: Feature importance analysis
        logger.info("Step 5: Analyzing feature importance...")
        feature_importance = model_trainer.get_feature_importance()
        
        if feature_importance is not None:
            logger.info("Top 10 most important features:")
            feature_names = feature_extractor.get_feature_names()
            if len(feature_names) == len(feature_importance):
                # Sort features by importance
                importance_indices = np.argsort(feature_importance)[::-1]
                for i, idx in enumerate(importance_indices[:10]):
                    logger.info(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
        
        # Step 6: Save training summary
        logger.info("Step 6: Saving training summary...")
        training_summary = {
            'total_samples': len(file_paths),
            'feature_dimensions': features.shape[1],
            'class_distribution': dict(zip(unique_labels, counts)),
            'training_time': time.time() - start_time,
            'models_trained': list(evaluation_results.keys()) if evaluation_results else [],
            'best_model': model_trainer.get_best_model_name() if hasattr(model_trainer, 'get_best_model_name') else 'ensemble'
        }
        
        # Save summary to file
        import json
        summary_path = os.path.join(config.models_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        logger.info(f"Training summary saved to: {summary_path}")
        
        total_time = time.time() - start_time
        logger.info(f"\nAdvanced training pipeline completed successfully in {total_time:.2f} seconds!")
        logger.info(f"Models saved in: {config.models_dir}")
        
        # Recommendations
        logger.info("\n" + "="*50)
        logger.info("TRAINING RECOMMENDATIONS:")
        logger.info("="*50)
        logger.info("1. Use the ensemble model for best overall performance")
        logger.info("2. Monitor per-class performance for class imbalance issues")
        logger.info("3. Consider data augmentation if accuracy plateaus")
        logger.info("4. Validate on independent test set before production")
        logger.info("5. Use advanced_predict.py for inference")
        
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
