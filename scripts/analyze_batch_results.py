#!/usr/bin/env python
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from datetime import datetime

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger

class BatchResultAnalyzer:
    """Analyzer cho batch test results"""
    
    def __init__(self, report_file):
        """Initialize với report file"""
        with open(report_file, 'r', encoding='utf-8') as f:
            self.report = json.load(f)
        
        self.batch_summary = self.report['batch_summary']
        self.quality_metrics = self.report['quality_metrics']
        self.component_reliability = self.report['component_reliability']
        self.individual_results = self.report['individual_results']
    
    def print_comprehensive_analysis(self):
        """In phân tích toàn diện"""
        print("="*80)
        print("📊 COMPREHENSIVE BATCH TEST ANALYSIS")
        print("="*80)
        
        # 1. Overall Performance
        self._analyze_overall_performance()
        
        # 2. Quality Metrics
        self._analyze_quality_metrics()
        
        # 3. Component Reliability
        self._analyze_component_reliability()
        
        # 4. Performance Distribution
        self._analyze_performance_distribution()
        
        # 5. Failure Analysis
        self._analyze_failures()
        
        # 6. Stability Assessment
        self._assess_stability()
        
        # 7. Production Readiness
        self._assess_production_readiness()
    
    def _analyze_overall_performance(self):
        """Phân tích performance tổng thể"""
        print(f"\n🚀 OVERALL PERFORMANCE")
        print("-" * 40)
        
        total = self.batch_summary['total_samples']
        successful = self.batch_summary['successful_samples']
        failed = self.batch_summary['failed_samples']
        success_rate = self.batch_summary['success_rate']
        avg_time = self.batch_summary['average_processing_time']
        total_time = self.batch_summary['total_batch_time']
        
        print(f"Total Samples: {total}")
        print(f"Successful: {successful} ({success_rate:.1f}%)")
        print(f"Failed: {failed} ({100-success_rate:.1f}%)")
        print(f"Average Processing Time: {avg_time:.1f}s per sample")
        print(f"Total Batch Time: {total_time:.1f}s")
        print(f"Throughput: {total/total_time*3600:.1f} samples/hour")
        
        # Performance assessment
        if success_rate >= 95:
            print("🏆 EXCELLENT performance - Production ready!")
        elif success_rate >= 85:
            print("✅ GOOD performance - Minor issues only")
        elif success_rate >= 70:
            print("⚠️ ACCEPTABLE performance - Improvements needed")
        else:
            print("❌ POOR performance - Major fixes required")
    
    def _analyze_quality_metrics(self):
        """Phân tích quality metrics"""
        print(f"\n🎯 QUALITY METRICS")
        print("-" * 40)
        
        avg_confidence = self.quality_metrics['average_confidence']
        avg_reformulation = self.quality_metrics['average_reformulation_quality']
        conf_range = self.quality_metrics['confidence_range']
        reform_range = self.quality_metrics['reformulation_range']
        
        print(f"Average Reasoning Confidence: {avg_confidence:.3f}")
        print(f"Confidence Range: {conf_range[0]:.3f} - {conf_range[1]:.3f}")
        
        if avg_confidence >= 0.8:
            print("🔥 EXCELLENT confidence levels")
        elif avg_confidence >= 0.6:
            print("✅ GOOD confidence levels")
        elif avg_confidence >= 0.4:
            print("⚠️ MODERATE confidence levels")
        else:
            print("❌ LOW confidence levels - investigate")
        
        print(f"\nAverage Reformulation Quality: {avg_reformulation:.3f}")
        print(f"Reformulation Range: {reform_range[0]:.3f} - {reform_range[1]:.3f}")
        
        if avg_reformulation >= 0.9:
            print("🔥 EXCELLENT query reformulation")
        elif avg_reformulation >= 0.7:
            print("✅ GOOD query reformulation")
        else:
            print("⚠️ Query reformulation needs improvement")
    
    def _analyze_component_reliability(self):
        """Phân tích component reliability"""
        print(f"\n🔧 COMPONENT RELIABILITY")
        print("-" * 40)
        
        total_samples = self.batch_summary['total_samples']
        
        for component, failures in self.component_reliability.items():
            reliability = ((total_samples - failures) / total_samples) * 100
            
            if reliability >= 95:
                status = "🏆 EXCELLENT"
            elif reliability >= 80:
                status = "✅ GOOD"
            elif reliability >= 60:
                status = "⚠️ MODERATE"
            else:
                status = "❌ POOR"
            
            print(f"{status} {component.upper()}: {reliability:.1f}% ({failures} failures)")
        
        # Identify most problematic components
        sorted_failures = sorted(self.component_reliability.items(), key=lambda x: x[1], reverse=True)
        if sorted_failures[0][1] > 0:
            print(f"\n⚠️ Most problematic component: {sorted_failures[0][0]} ({sorted_failures[0][1]} failures)")
    
    def _analyze_performance_distribution(self):
        """Phân tích phân phối performance"""
        print(f"\n📈 PERFORMANCE DISTRIBUTION")
        print("-" * 40)
        
        # Analyze processing times from individual results
        processing_times = [r['processing_time'] for r in self.individual_results if r['success']]
        
        if processing_times:
            min_time = min(processing_times)
            max_time = max(processing_times)
            median_time = np.median(processing_times)
            std_time = np.std(processing_times)
            
            print(f"Processing Time Statistics:")
            print(f"  Min: {min_time:.1f}s")
            print(f"  Max: {max_time:.1f}s")
            print(f"  Median: {median_time:.1f}s")
            print(f"  Std Dev: {std_time:.1f}s")
            
            # Consistency assessment
            cv = std_time / np.mean(processing_times)  # Coefficient of variation
            if cv < 0.2:
                print("🎯 CONSISTENT processing times")
            elif cv < 0.5:
                print("✅ REASONABLY consistent processing times")
            else:
                print("⚠️ HIGH variability in processing times")
    
    def _analyze_failures(self):
        """Phân tích failures chi tiết"""
        print(f"\n❌ FAILURE ANALYSIS")
        print("-" * 40)
        
        failed_results = [r for r in self.individual_results if not r['success']]
        
        if not failed_results:
            print("🎉 NO FAILURES - Perfect batch!")
            return
        
        print(f"Total Failures: {len(failed_results)}")
        
        # Analyze failure patterns
        failure_types = {}
        for result in failed_results:
            error = result.get('error', 'Unknown error')
            error_type = error.split(':')[0] if ':' in error else error
            failure_types[error_type] = failure_types.get(error_type, 0) + 1
        
        print(f"\nFailure Types:")
        for error_type, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count} occurrences")
        
        # Sample IDs that failed
        failed_ids = [r['sample_id'] for r in failed_results]
        print(f"\nFailed Samples: {', '.join(failed_ids[:5])}")
        if len(failed_ids) > 5:
            print(f"  ... and {len(failed_ids)-5} more")
    
    def _assess_stability(self):
        """Đánh giá stability của system"""
        print(f"\n🏗️ SYSTEM STABILITY ASSESSMENT")
        print("-" * 40)
        
        success_rate = self.batch_summary['success_rate']
        total_component_failures = sum(self.component_reliability.values())
        total_samples = self.batch_summary['total_samples']
        
        # Stability score calculation
        stability_factors = {
            'success_rate': success_rate / 100,
            'component_reliability': 1 - (total_component_failures / (total_samples * len(self.component_reliability))),
            'processing_consistency': self._calculate_processing_consistency()
        }
        
        overall_stability = np.mean(list(stability_factors.values()))
        
        print(f"Stability Factors:")
        for factor, score in stability_factors.items():
            print(f"  {factor.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\nOverall Stability Score: {overall_stability:.3f}")
        
        if overall_stability >= 0.9:
            print("🏆 HIGHLY STABLE - Production ready")
        elif overall_stability >= 0.8:
            print("✅ STABLE - Minor monitoring needed")
        elif overall_stability >= 0.7:
            print("⚠️ MODERATELY STABLE - Improvements recommended")
        else:
            print("❌ UNSTABLE - Major fixes required")
    
    def _calculate_processing_consistency(self):
        """Tính processing consistency"""
        processing_times = [r['processing_time'] for r in self.individual_results if r['success']]
        
        if len(processing_times) < 2:
            return 1.0
        
        cv = np.std(processing_times) / np.mean(processing_times)
        # Convert CV to consistency score (lower CV = higher consistency)
        consistency = max(0, 1 - cv)
        return consistency
    
    def _assess_production_readiness(self):
        """Đánh giá production readiness"""
        print(f"\n🚀 PRODUCTION READINESS ASSESSMENT")
        print("-" * 40)
        
        # Production readiness criteria
        criteria = {
            'Success Rate ≥ 90%': self.batch_summary['success_rate'] >= 90,
            'Average Confidence ≥ 0.6': self.quality_metrics['average_confidence'] >= 0.6,
            'Average Processing Time ≤ 30s': self.batch_summary['average_processing_time'] <= 30,
            'No Critical Component Failures': max(self.component_reliability.values()) <= 1,
            'Reformulation Quality ≥ 0.7': self.quality_metrics['average_reformulation_quality'] >= 0.7
        }
        
        passed_criteria = sum(criteria.values())
        total_criteria = len(criteria)
        
        print(f"Production Readiness Criteria:")
        for criterion, passed in criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} {criterion}")
        
        readiness_score = passed_criteria / total_criteria
        print(f"\nProduction Readiness Score: {passed_criteria}/{total_criteria} ({readiness_score*100:.1f}%)")
        
        if readiness_score >= 0.9:
            print("🚀 PRODUCTION READY - Deploy with confidence!")
        elif readiness_score >= 0.7:
            print("✅ NEARLY READY - Minor improvements recommended")
        elif readiness_score >= 0.5:
            print("⚠️ NEEDS WORK - Address failing criteria")
        else:
            print("❌ NOT READY - Significant improvements required")
    
    def create_visualizations(self, output_dir):
        """Tạo visualizations cho batch results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Success Rate Pie Chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success/Failure distribution
        success_data = [self.batch_summary['successful_samples'], self.batch_summary['failed_samples']]
        success_labels = ['Successful', 'Failed']
        colors = ['#2ecc71', '#e74c3c']
        
        ax1.pie(success_data, labels=success_labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Success Rate Distribution')
        
        # Component Reliability
        components = list(self.component_reliability.keys())
        failures = list(self.component_reliability.values())
        total_samples = self.batch_summary['total_samples']
        reliabilities = [((total_samples - f) / total_samples) * 100 for f in failures]
        
        bars = ax2.bar(components, reliabilities, color='skyblue')
        ax2.set_title('Component Reliability (%)')
        ax2.set_ylabel('Reliability %')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rel in zip(bars, reliabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rel:.1f}%', ha='center', va='bottom')
        
        # Processing Time Distribution
        processing_times = [r['processing_time'] for r in self.individual_results if r['success']]
        if processing_times:
            ax3.hist(processing_times, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
            ax3.set_title('Processing Time Distribution')
            ax3.set_xlabel('Processing Time (seconds)')
            ax3.set_ylabel('Frequency')
            ax3.axvline(np.mean(processing_times), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(processing_times):.1f}s')
            ax3.legend()
        
        # Quality Metrics
        metrics = ['Confidence', 'Reformulation']
        values = [self.quality_metrics['average_confidence'], 
                 self.quality_metrics['average_reformulation_quality']]
        
        bars = ax4.bar(metrics, values, color=['orange', 'purple'])
        ax4.set_title('Average Quality Metrics')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'batch_analysis_overview.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to {output_dir}/batch_analysis_overview.png")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Batch Test Results')
    parser.add_argument('--report-file', type=str, required=True, 
                       help='Path to batch test report JSON file')
    parser.add_argument('--output-dir', type=str, default='data/batch_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--create-visualizations', action='store_true',
                       help='Create visualization charts')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.report_file):
        print(f"❌ Report file not found: {args.report_file}")
        return
    
    # Initialize analyzer
    analyzer = BatchResultAnalyzer(args.report_file)
    
    # Run comprehensive analysis
    analyzer.print_comprehensive_analysis()
    
    # Create visualizations if requested
    if args.create_visualizations:
        analyzer.create_visualizations(args.output_dir)
    
    print(f"\n📁 Analysis completed for: {args.report_file}")

if __name__ == "__main__":
    main()
