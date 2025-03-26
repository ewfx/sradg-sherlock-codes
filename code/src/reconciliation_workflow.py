from data_processor import DataProcessor
from anomaly_detector import AnomalyDetector
from anomaly_classifier import HybridAnomalyClassifier
from ai_insights import AIIInsightsGenerator

class ReconciliationWorkflow:
    """Orchestrates the complete reconciliation process with HuggingFace"""
    
    def __init__(self, input_path, output_path, hf_token=None):
        self.input_path = input_path
        self.output_path = output_path
        self.hf_token = hf_token
        self.results = None
    
    def execute(self):
        """Execute the complete workflow"""
        try:
            # 1. Data Processing
            print("Processing data...")
            processor = DataProcessor(self.input_path)
            df = processor.full_pipeline()
            
            # 2. Anomaly Detection
            print("Detecting anomalies...")
            detector = AnomalyDetector()
            df = detector.detect_anomalies(df)
            df = detector.calculate_anomaly_scores(df)
            
            # 3. Break Classification
            print("Classifying breaks...")
            classifier = HybridAnomalyClassifier()
            df = classifier.classify_all(df)
            
            # 4. AI Insights with HuggingFace
            print("Generating insights with HuggingFace...")
            ai = AIIInsightsGenerator(hf_token=self.hf_token)
            df = ai.generate_break_comments(df)
            summary = ai.generate_executive_summary(df)
            
            # 5. Save Results
            print("Saving results...")
            self._save_results(df, summary)
            
            print("\n=== PROCESSING COMPLETE ===")
            return True
            
        except Exception as e:
            print(f"\n!!! PROCESSING FAILED: {str(e)}")
            return False
    
    def _save_results(self, df, summary):
        """Save results to output file"""
        # Save detailed results
        df.to_excel(self.output_path, index=False)
        
        # Save summary to text file
        summary_path = self.output_path.replace('.xlsx', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        self.results = df
        print(f"\nResults saved to: {self.output_path}")
        print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "SAMPLE_DATA2.xlsx"
    OUTPUT_FILE = "RECONCILIATION_RESULTS_HF.xlsx"
    HF_TOKEN = None  # Replace with your HuggingFace token
    
    # Execute workflow
    workflow = ReconciliationWorkflow(INPUT_FILE, OUTPUT_FILE, HF_TOKEN)
    success = workflow.execute()
    
    if success:
        print("\nReconciliation completed successfully using HuggingFace!")
    else:
        print("\nReconciliation failed. Check error messages above.")