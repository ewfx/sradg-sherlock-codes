from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import os

class AIIInsightsGenerator:
    """Generates AI-powered insights using HuggingFace models"""
    
    def __init__(self, hf_token=None, model="google/flan-t5-large"):
        self.hf_token = hf_token
        self.model_name = model
        self.llm_chain = self._initialize_llm_chain()
        
        # Define templates
        self.templates = {
            'break_comment': """
            Analyze this accounting reconciliation break:
            Date: {asof_date}
            Account: {account_details}
            GL Balance: {gl_balance}
            IHUB Balance: {ihub_balance}
            Difference: {balance_diff}
            Previous Difference: {prev_diff}
            Classification: {classification}
            
            Possible reasons for this discrepancy:
            """,
            
            'executive_summary': """
            Summarize these reconciliation results:
            Total Records: {total_records}
            Matches: {matches} ({match_pct:.1f}%)
            Breaks: {breaks} ({break_pct:.1f}%)
            Anomalies: {anomalies}
            
            Key observations:
            """
        }
    
    def _initialize_llm_chain(self):
        """Initialize the LLM chain with HuggingFace model"""
        if not self.hf_token:
            return None
            
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = self.hf_token
        
        # Initialize HuggingFace model
        llm = HuggingFaceHub(
            repo_id=self.model_name,
            model_kwargs={"temperature": 0.2, "max_length": 200}
        )
        
        return llm
    
    def _generate_with_llm(self, prompt_text):
        """Generate text using the LLM"""
        if not self.llm_chain:
            return "Analysis unavailable (no HuggingFace token provided)"
            
        try:
            return self.llm_chain(prompt_text)
        except Exception as e:
            print(f"LLM generation failed: {str(e)}")
            return "Analysis generation failed"
    
    def generate_break_comments(self, df):
        """Generate comments for all break records"""
        if self.llm_chain is None:
            df['Comments'] = np.where(
                df['Match Status'] == 'Break',
                'Discrepancy detected - requires investigation',
                'Difference within acceptable tolerance'
            )
            return df
            
        breaks_df = df[df['Match Status'] == 'Break'].copy()
        if breaks_df.empty:
            return df
            
        # Generate comments
        comments = []
        for _, row in breaks_df.iterrows():
            prompt = self.templates['break_comment'].format(
                asof_date=row['AsofDate'].strftime('%Y-%m-%d'),
                account_details=f"{row['Company']}-{row['Account']} ({row['Currency']})",
                gl_balance=row['GL Balance'],
                ihub_balance=row['IHUB Balance'],
                balance_diff=row['Balance Difference'],
                prev_diff=row.get('Previous Balance Difference', 'N/A'),
                classification=row.get('Break Classification', 'Unknown')
            )
            response = self._generate_with_llm(prompt)
            comments.append(response)
        
        breaks_df['Comments'] = comments
        df.update(breaks_df)
        return df
    
    def generate_executive_summary(self, df):
        """Generate an executive summary report"""
        if self.llm_chain is None:
            return "AI summary unavailable (no HuggingFace token provided)"
            
        # Calculate metrics
        total_records = len(df)
        matches = len(df[df['Match Status'] == 'Match'])
        breaks = total_records - matches
        anomalies = df['Is Anomaly'].sum()
        
        # Generate summary
        prompt = self.templates['executive_summary'].format(
            total_records=total_records,
            matches=matches,
            match_pct=(matches/total_records)*100,
            breaks=breaks,
            break_pct=(breaks/total_records)*100,
            anomalies=anomalies
        )
        
        return self._generate_with_llm(prompt)