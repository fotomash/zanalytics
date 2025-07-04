import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List  # ← ADD THIS
import warnings

warnings.filterwarnings('ignore')

from typing import Dict, Optional
# 1. Real-time Regime Change Alerts
def setup_realtime_alerts(self):
    """Initialize real-time alert system"""
    self.last_regime = None
    self.alert_queue = []
    
def check_regime_alerts(self, current_regime: str, df: pd.DataFrame):
    """Check for regime changes and trigger alerts"""
    if self.last_regime and current_regime != self.last_regime:
        # Regime change detected
        alert = {
            'type': 'regime_change',
            'from': self.last_regime,
            'to': current_regime,
            'timestamp': df['timestamp'].iloc[-1],
            'vpin': df['vpin'].iloc[-1],
            'confidence': 0.9
        }
        
        # Show toast notification
        st.toast(f"⚠️ Regime Change: {self.last_regime} → {current_regime}", icon='🔄')
        
        # Add to sidebar alerts
        with st.sidebar:
            st.error(f"REGIME ALERT: Now in {current_regime.upper()} regime")
            
        self.alert_queue.append(alert)
        
    self.last_regime = current_regime

# 2. Automated Pattern Snapshots with Download
def create_pattern_snapshot(self, df: pd.DataFrame, events: Dict) -> Dict:
    """Create comprehensive snapshot of all detected patterns"""
    snapshot = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.current_symbol,
            'tick_count': len(df),
            'regime': self.session_state['market_regime']
        },
        'events': {
            'iceberg': self.session_state['iceberg_events'],
            'spoofing': self.session_state['spoofing_events'],
            'layering': events.get('layering_events', []),
            'quote_stuffing': self.session_state.get('quote_stuffing_events', []),
            'liquidity_sweeps': events.get('sweep_events', []),
            'micro_wyckoff': events.get('wyckoff_events', {}),
            'inducement_traps': events.get('trap_events', [])
        },
        'statistics': {
            'manipulation_score': self.session_state['manipulation_score'],
            'avg_vpin': df['vpin'].mean(),
            'regime_distribution': df['regime_simple'].value_counts().to_dict()
        }
    }
    return snapshot

def render_download_buttons(self, snapshot: Dict):
    """Render download buttons for pattern exports"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv_data = self.snapshot_to_csv(snapshot)
        st.download_button(
            label="📊 Download Events CSV",
            data=csv_data,
            file_name=f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON export
        json_data = json.dumps(snapshot, indent=2, default=str)
        st.download_button(
            label="📋 Download Events JSON",
            data=json_data,
            file_name=f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# 3. Deep Pattern Search with Natural Language
def create_pattern_search_interface(self):
    """Create advanced pattern search interface"""
    st.markdown("### 🔍 Deep Pattern Search")
    
    search_query = st.text_input(
        "Search patterns (e.g., 'sweeps > 15bps between 09:00-11:00')",
        key="pattern_search"
    )
    
    if search_query:
        results = self.execute_pattern_search(search_query)
        if results:
            st.dataframe(pd.DataFrame(results))
        else:
            st.info("No patterns match your search criteria")

def execute_pattern_search(self, query: str) -> List[Dict]:
    """Execute natural language pattern search"""
    # Parse query for filters
    filters = self.parse_search_query(query)
    
    # Apply filters to events
    all_events = []
    
    # Collect all events with metadata
    for event in self.session_state.get('sweep_events', []):
        event['event_type'] = 'sweep'
        all_events.append(event)
        
    # Filter based on parsed criteria
    filtered = all_events
    
    if 'magnitude_min' in filters:
        filtered = [e for e in filtered if e.get('magnitude_bps', 0) > filters['magnitude_min']]
        
    if 'time_range' in filters:
        start, end = filters['time_range']
        filtered = [e for e in filtered if start <= e['timestamp'].time() <= end]
        
    return filtered

# 4. ML-Based Trap Prediction
def create_trap_predictor(self):
    """Initialize ML model for trap prediction"""
    from sklearn.ensemble import GradientBoostingClassifier
    
    self.trap_predictor = GradientBoostingClassifier(
        n_estimators=100,
        learning_depth=3,
        random_state=42
    )

def predict_next_trap(self, df: pd.DataFrame) -> Dict:
    """Predict likelihood of trap in next N ticks"""
    # Extract features
    features = self.extract_ml_features(df.tail(100))
    
    # Get prediction
    trap_probability = self.trap_predictor.predict_proba(features)[0, 1]
    
    return {
        'trap_probability': trap_probability,
        'confidence': self.calculate_prediction_confidence(features),
        'suggested_action': 'WAIT' if trap_probability > 0.7 else 'PROCEED',
        'key_indicators': self.get_top_features(features)
    }

# 5. Event Correlation Matrix
def create_event_correlation_matrix(self, df: pd.DataFrame):
    """Create correlation matrix between different manipulation events"""
    # Create binary event series
    event_matrix = pd.DataFrame(index=df.index)
    
    # Mark iceberg events
    for event in self.session_state['iceberg_events']:
        mask = (df['timestamp'] >= event['start_time']) & (df['timestamp'] <= event['end_time'])
        event_matrix.loc[mask, 'iceberg'] = 1
    event_matrix['iceberg'].fillna(0, inplace=True)
    
    # Mark spoofing events
    for event in self.session_state['spoofing_events']:
        idx = df[df['timestamp'] == event['timestamp']].index
        if not idx.empty:
            event_matrix.loc[idx[0], 'spoofing'] = 1
    event_matrix['spoofing'].fillna(0, inplace=True)
    
    # Calculate correlations
    corr_matrix = event_matrix.corr()
    
    # Visualize
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(title="Event Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detect causal chains
    chains = self.detect_causal_chains(event_matrix)
    if chains:
        st.markdown("#### 🔗 Detected Causal Chains")
        for chain in chains:
            st.write(f"• {' → '.join(chain['sequence'])}: {chain['confidence']:.2%} confidence")

# 6. Session Summary with Auto-Narrative
def generate_session_narrative(self, df: pd.DataFrame, events: Dict) -> str:
    """Generate human-readable session summary"""
    narrative = []
    
    # Opening
    narrative.append(f"### Session Analysis: {df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')} - {df['timestamp'].iloc[-1].strftime('%H:%M')}")
    
    # Market regime
    regime_dist = df['regime_simple'].value_counts(normalize=True)
    dominant_regime = regime_dist.index[0]
    narrative.append(f"\n**Market Regime**: Predominantly {dominant_regime} ({regime_dist[dominant_regime]:.1%} of session)")
    
    # Key events
    total_events = (
        len(self.session_state['iceberg_events']) +
        len(self.session_state['spoofing_events']) +
        len(events.get('sweep_events', []))
    )
    narrative.append(f"\n**Manipulation Events**: {total_events} total events detected")
    
    # Price action
    price_change = df['price_mid'].iloc[-1] - df['price_mid'].iloc[0]
    price_change_pct = (price_change / df['price_mid'].iloc[0]) * 10000  # in bps
    narrative.append(f"\n**Price Action**: {price_change_pct:+.1f} bps move")
    
    # Toxicity
    avg_vpin = df['vpin'].mean()
    toxic_periods = (df['vpin'] > 0.7).sum() / len(df) * 100
    narrative.append(f"\n**Flow Toxicity**: Average VPIN {avg_vpin:.3f}, {toxic_periods:.1f}% toxic periods")
    
    # Recommendations
    if dominant_regime == 'manipulated' or toxic_periods > 30:
        narrative.append("\n⚠️ **Recommendation**: Exercise caution - high manipulation/toxicity detected. Consider wider stops and reduced position sizing.")
    
    return '\n'.join(narrative)

# 7. PDF Report Generation
def generate_pdf_report(self, df: pd.DataFrame, snapshot: Dict, narrative: str):
    """Generate comprehensive PDF report"""
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
    from reportlab.lib.styles import getSampleStyleSheet
    
    # Create PDF
    pdf_path = f"quantum_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    
    # Build content
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    story.append(Paragraph("Quantum Microstructure Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Narrative
    story.append(Paragraph(narrative, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Event summary table
    event_data = [
        ['Event Type', 'Count', 'Avg Confidence'],
        ['Iceberg Orders', len(snapshot['events']['iceberg']), 
         f"{np.mean([e['confidence'] for e in snapshot['events']['iceberg']]):.2f}" if snapshot['events']['iceberg'] else 'N/A'],
        ['Spoofing', len(snapshot['events']['spoofing']),
         f"{np.mean([e['confidence'] for e in snapshot['events']['spoofing']]):.2f}" if snapshot['events']['spoofing'] else 'N/A'],
        ['Liquidity Sweeps', len(snapshot['events']['liquidity_sweeps']),
         f"{np.mean([e['confidence'] for e in snapshot['events']['liquidity_sweeps']]):.2f}" if snapshot['events']['liquidity_sweeps'] else 'N/A']
    ]
    
    event_table = Table(event_data)
    story.append(event_table)
    
    # Build PDF
    doc.build(story)
    
    return pdf_path

# Integration into main dashboard
def enhance_dashboard_section(self):
    """Add enhanced features to dashboard"""
    
    # Real-time alerts section
    if self.config['enhanced_features']['realtime_alerts']['enabled']:
        self.check_regime_alerts(self.session_state['market_regime'], df)
    
    # Pattern search
    if self.config['enhanced_features']['pattern_search']['enabled']:
        with st.expander("🔍 Advanced Pattern Search", expanded=False):
            self.create_pattern_search_interface()
    
    # ML predictions
    if self.config['enhanced_features']['ml_prediction']['enabled']:
        st.markdown("### 🤖 ML-Based Predictions")
        col1, col2 = st.columns(2)
        
        with col1:
            trap_pred = self.predict_next_trap(df)
            st.metric(
                "Trap Probability (Next 50 ticks)",
                f"{trap_pred['trap_probability']:.1%}",
                delta=trap_pred['suggested_action']
            )
        
        with col2:
            regime_pred = self.predict_next_regime(df)
            st.metric(
                "Predicted Regime Change",
                regime_pred['next_regime'],
                delta=f"in ~{regime_pred['ticks_until_change']} ticks"
            )
    
    # Event correlation
    with st.expander("🔗 Event Correlation Analysis", expanded=False):
        self.create_event_correlation_matrix(df)
    
    # Session summary & export
    st.markdown("### 📊 Session Summary & Export")
    
    # Generate narrative
    narrative = self.generate_session_narrative(df, all_events)
    st.markdown(narrative)
    
    # Export buttons
    snapshot = self.create_pattern_snapshot(df, all_events)
    self.render_download_buttons(snapshot)
    
    # PDF generation
    if st.button("📄 Generate PDF Report"):
        with st.spinner("Generating PDF report..."):
            pdf_path = self.generate_pdf_report(df, snapshot, narrative)
            st.success(f"PDF report generated: {pdf_path}")
if __name__ == "__main__":
    # Option 1: If your dashboard is a class, instantiate and call main
    # dash = MyDashboardClass()   # whatever your main class is called
    # dash.run()                 # whatever your main run function is

    # Option 2: If your dashboard is procedural, call your main function
    # main()                     # If you have a main() defined

    # Option 3: For quick test, add something like:
    st.title("SMC & Wyckoff Dashboard")
    st.write("The dashboard is loaded. Now implement the logic in a main() function.")