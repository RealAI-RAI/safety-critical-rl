import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import torch
import json

# Set page config
st.set_page_config(
    page_title="Safe-RL Training Dashboard",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2ecc71;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

class TrainingDashboard:
    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        self.data = self.load_data()
    
    def load_data(self):
        """Load all training data"""
        data = {}
        
        # Load checkpoints
        checkpoints = sorted(list(self.experiment_dir.glob("checkpoint_*.pt")))
        episodes, rewards, violations = [], [], []
        
        for cp in checkpoints:
            try:
                checkpoint = torch.load(cp, map_location='cpu')
                episodes.append(checkpoint['episode'])
                rewards.append(checkpoint['episode_reward'])
                
                # Sum all violations
                v = checkpoint.get('violations', {})
                total_violations = sum(v.values()) if v else 0
                violations.append(total_violations)
            except:
                continue
        
        data['episodes'] = episodes
        data['rewards'] = rewards
        data['violations'] = violations
        
        # Load summary if exists
        summary_path = self.experiment_dir / "training_summary.csv"
        if summary_path.exists():
            data['summary'] = pd.read_csv(summary_path)
        
        return data
    
    def create_dashboard(self):
        """Create Streamlit dashboard"""
        
        # Header
        st.markdown('<h1 class="main-header">🚄 Safe-RL Training Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("📊 Dashboard Controls")
            
            experiment_name = st.text_input(
                "Experiment Directory",
                value=str(self.experiment_dir)
            )
            
            st.header("📈 Visualization Options")
            show_rewards = st.checkbox("Show Rewards", value=True)
            show_violations = st.checkbox("Show Violations", value=True)
            show_summary = st.checkbox("Show Summary", value=True)
            
            smoothing_window = st.slider(
                "Smoothing Window",
                min_value=1,
                max_value=50,
                value=10
            )
            
        # Main content
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Episodes</div>
            </div>
            """.format(len(self.data['episodes'])), unsafe_allow_html=True)
        
        with col2:
            if self.data['rewards']:
                avg_reward = sum(self.data['rewards']) / len(self.data['rewards'])
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{:.2f}</div>
                    <div class="metric-label">Average Reward</div>
                </div>
                """.format(avg_reward), unsafe_allow_html=True)
        
        with col3:
            if self.data['violations']:
                total_violations = sum(self.data['violations'])
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Total Violations</div>
                </div>
                """.format(total_violations), unsafe_allow_html=True)
        
        with col4:
            if self.data['rewards']:
                final_reward = self.data['rewards'][-1]
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{:.2f}</div>
                    <div class="metric-label">Final Reward</div>
                </div>
                """.format(final_reward), unsafe_allow_html=True)
        
        # Charts
        if show_rewards and self.data['episodes']:
            st.header("📈 Training Progress")
            
            tab1, tab2, tab3 = st.tabs(["Raw Data", "Smoothed", "Comparison"])
            
            with tab1:
                # Raw rewards
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=self.data['episodes'],
                    y=self.data['rewards'],
                    mode='lines+markers',
                    name='Reward',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title='Episode Rewards (Raw)',
                    xaxis_title='Episode',
                    yaxis_title='Reward',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Smoothed rewards
                df = pd.DataFrame({
                    'Episode': self.data['episodes'],
                    'Reward': self.data['rewards']
                })
                df['Smoothed'] = df['Reward'].rolling(
                    window=smoothing_window, center=True
                ).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Episode'],
                    y=df['Reward'],
                    mode='markers',
                    name='Raw',
                    marker=dict(color='lightblue', size=4),
                    opacity=0.5
                ))
                fig.add_trace(go.Scatter(
                    x=df['Episode'],
                    y=df['Smoothed'],
                    mode='lines',
                    name='Smoothed',
                    line=dict(color='darkblue', width=3)
                ))
                fig.update_layout(
                    title=f'Episode Rewards (Smoothed, window={smoothing_window})',
                    xaxis_title='Episode',
                    yaxis_title='Reward',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Violations vs Rewards
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=self.data['episodes'],
                    y=self.data['rewards'],
                    mode='lines',
                    name='Reward',
                    yaxis='y1',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=self.data['episodes'],
                    y=self.data['violations'],
                    mode='lines',
                    name='Violations',
                    yaxis='y2',
                    line=dict(color='red', width=2)
                ))
                fig.update_layout(
                    title='Reward vs Violations',
                    xaxis_title='Episode',
                    yaxis=dict(title='Reward', side='left'),
                    yaxis2=dict(
                        title='Violations',
                        side='right',
                        overlaying='y'
                    ),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        if show_summary and 'summary' in self.data:
            st.header("📋 Training Summary")
            st.dataframe(self.data['summary'], use_container_width=True)
            
            # Export options
            st.download_button(
                label="📥 Download Summary as CSV",
                data=self.data['summary'].to_csv(index=False),
                file_name="training_summary.csv",
                mime="text/csv"
            )
        
        # File browser
        st.header("📁 Experiment Files")
        
        files = list(self.experiment_dir.glob("*"))
        file_info = []
        
        for file in files:
            size = file.stat().st_size / 1024  # KB
            file_info.append({
                'Filename': file.name,
                'Size (KB)': f"{size:.1f}",
                'Type': file.suffix if file.is_file() else 'Directory',
                'Last Modified': file.stat().st_mtime
            })
        
        if file_info:
            files_df = pd.DataFrame(file_info)
            files_df = files_df.sort_values('Last Modified', ascending=False)
            st.dataframe(files_df[['Filename', 'Size (KB)', 'Type']], 
                        use_container_width=True)
        
        # Download section
        st.header("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Full Report"):
                # This would generate and download a comprehensive report
                st.info("Report generation feature coming soon!")
        
        with col2:
            if st.button("🖼️ Save All Plots"):
                st.info("Plot saving feature coming soon!")
        
        with col3:
            if st.button("📈 Export Data"):
                # Export all data as JSON
                export_data = {
                    'episodes': self.data['episodes'],
                    'rewards': self.data['rewards'],
                    'violations': self.data['violations']
                }
                
                import json
                json_str = json.dumps(export_data, indent=2)
                
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="training_data.json",
                    mime="application/json"
                )

# Main app
def main():
    st.title("Safe Reinforcement Learning Training Monitor")
    
    # Default experiment directory