import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils import FEATURES_CSV_PATH, FIGURES_DIR

df = pd.read_csv(FEATURES_CSV_PATH)

def plot_dataset_overview(dataframe):
    '''
    Plots data using features extracted from entire dataset, shows an overview of the dataset.
    Plots:
     - Number of images by diagnosis
     - Image distribution in %
     - Images per patient distribution
     - How many patients per diagnosis (Patient-Level diagnosis distribution)
    
    '''
    
    df = dataframe
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Number of images by diagnosis
    df['diagnostic'].value_counts().plot(kind='bar', ax=axes[0,0], color=['#2ecc71', '#e74c3c'])
    axes[0,0].set_title('Number of images by diagnosis', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Diagnosis')
    axes[0,0].set_ylabel('Number of Images')
    for i, v in enumerate(df['diagnostic'].value_counts().values):
        axes[0,0].text(i, v + 5, str(v), ha='center', fontweight='bold')

    # Image distribution in %
    percentages = df['diagnostic'].value_counts(normalize=True) * 100
    axes[0,1].pie(percentages, labels=percentages.index, autopct='%1.1f%%', 
                colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0,1].set_title('Image Distribution (%)', fontsize=12, fontweight='bold')

    # Images per patient distribution
    images_per_patient = df.groupby('patient_id').size()
    axes[1,0].hist(images_per_patient, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1,0].axvline(images_per_patient.mean(), color='red', linestyle='--', 
                    label=f'Mean: {images_per_patient.mean():.1f}')
    axes[1,0].set_title('Images per Patient Distribution', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Number of Images per Patient')
    axes[1,0].set_ylabel('Number of Patients')
    axes[1,0].legend()

    # Patient-level class distribution
    patient_diagnosis = df.groupby('patient_id')['diagnostic'].first().value_counts()
    axes[1,1].bar(patient_diagnosis.index, patient_diagnosis.values, color=['#2ecc71', '#e74c3c'])
    axes[1,1].set_title('Patient-Level Diagnosis Distribution', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Diagnosis')
    axes[1,1].set_ylabel('Number of Patients')
    for i, v in enumerate(patient_diagnosis.values):
        axes[1,1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

    plt.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'dataset-overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_abcd_features(dataframe, save=True):
    '''
    Plots all ABCD features: Asymmetry, Border, and Color (Hue, Saturation, Value)
    '''
    df = dataframe
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # A - Asymmetry
    ax = axes[0, 0]
    for diagnosis in df['diagnostic'].unique():
        data = df[df['diagnostic'] == diagnosis]['asymmetry_score']
        ax.hist(data, alpha=0.6, bins=20, label=diagnosis, density=True)
    ax.set_title('A: Asymmetry Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # B - Border
    ax = axes[0, 1]
    for diagnosis in df['diagnostic'].unique():
        data = df[df['diagnostic'] == diagnosis]['border_score']
        ax.hist(data, alpha=0.6, bins=20, label=diagnosis, density=True)
    ax.set_title('B: Border Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    
    # C - Hue (Color)
    ax = axes[0, 2]
    for diagnosis in df['diagnostic'].unique():
        data = df[df['diagnostic'] == diagnosis]['hue_mean']
        ax.hist(data, alpha=0.6, bins=20, label=diagnosis, density=True)
    ax.set_title('C: Hue Mean (Color)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hue Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # C - Saturation (Color Intensity)
    ax = axes[1, 0]
    for diagnosis in df['diagnostic'].unique():
        data = df[df['diagnostic'] == diagnosis]['saturation_mean']
        ax.hist(data, alpha=0.6, bins=20, label=diagnosis, density=True)
    ax.set_title('C: Saturation Mean (Color Intensity)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Saturation')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # C - Value (Brightness)
    ax = axes[1, 1]
    for diagnosis in df['diagnostic'].unique():
        data = df[df['diagnostic'] == diagnosis]['value_mean']
        ax.hist(data, alpha=0.6, bins=20, label=diagnosis, density=True)
    ax.set_title('C: Value Mean (Brightness)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Color variation (std of hue - how varied the colors are)
    ax = axes[1, 2]
    for diagnosis in df['diagnostic'].unique():
        data = df[df['diagnostic'] == diagnosis]['hue_std']
        ax.hist(data, alpha=0.6, bins=20, label=diagnosis, density=True)
    ax.set_title('C: Hue Std Dev (Color Variation)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Standard Deviation')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('ABCD Rule Features: Asymmetry, Border, and Color Components', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'abc-overview.png'), dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_color_features(dataframe):
    df = dataframe
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    color_features = ['hue_mean', 'saturation_mean', 'value_mean']
    titles = ['Hue (Color)', 'Saturation (Color Intensity)', 'Value (Brightness)']

    for idx, (feature, title) in enumerate(zip(color_features, titles)):
        for diagnosis in df['diagnostic'].unique():
            data = df[df['diagnostic'] == diagnosis][feature].dropna()
            axes[idx].hist(data, alpha=0.6, label=diagnosis, bins=25, density=True)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Color Feature Distributions by Diagnosis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'color-features.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(dataframe):
    df = dataframe
    # Select key features (avoid too many)
    key_features = ['asymmetry_score', 'border_score', 
                    'hue_mean', 'hue_std', 'hue_skew',
                    'saturation_mean', 'saturation_std', 
                    'value_mean', 'value_std',
                    'hue_5p', 'hue_50p', 'hue_95p']

    # Create numeric diagnosis column
    df['diagnostic_num'] = (df['diagnostic'] == 'malignant').astype(int)

    corr_matrix = df[key_features + ['diagnostic_num']].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, cbar_kws={"shrink": 0.8},
                annot_kws={'size': 8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature-correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print high correlations
    print("\nHigh correlations with diagnosis (>0.3 or <-0.3):")
    for feature in key_features:
        corr = corr_matrix.loc[feature, 'diagnostic_num']
        if abs(corr) > 0.3:
            print(f"  {feature}: {corr:.3f}")
            
def plot_feature_distribution(dataframe):
    '''Use boxplots instead of violin plots (more robust)'''
    
    top_features = ['asymmetry_score', 'border_score', 'hue_mean', 'value_mean']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(top_features):
        # Prepare data for boxplot
        benign_data = dataframe[dataframe['diagnostic'] == 'benign'][feature].dropna()
        malignant_data = dataframe[dataframe['diagnostic'] == 'malignant'][feature].dropna()
        
        # Create boxplot
        bp = axes[idx].boxplot([benign_data, malignant_data], 
                               labels=['Benign', 'Malignant'],
                               patch_artist=True,
                               showmeans=True,
                               meanline=True)
        
        # Color the boxes
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        
        axes[idx].set_title(f'{feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions: Benign vs Malignant', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature-distribution-boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

plot_feature_distribution(df)