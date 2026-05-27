import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import os
from utils import FEATURES_CSV_PATH, FIGURES_DIR, IMGS_DIR, METADATA_CSV_PATH



def plot_diagnosis_distribution(df, save=False, show=True):
    colors = ['#2ecc71' if diag in ['ACK', 'NEV', 'SEK'] else '#e74c3c' for diag in df['diagnostic'].value_counts().index]
    custom_lines = [Line2D([0], [0], color='#2ecc71', lw=4),
                Line2D([0], [0], color='#e74c3c', lw=4),]
    
    ax = df['diagnostic'].value_counts().plot(kind='bar', color=colors)
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    ax.legend(custom_lines, ['Disease', 'Cancer'])
    
    for i, v in enumerate(df['diagnostic'].value_counts().values):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'diag-dist.png'), dpi=300, bbox_inches='tight')
        plt.close()
    if show:
        plt.show()
        plt.close()
    
def plot_diagnosis_percentage_distribution(df, save=False, show=True):
    counts = df['diagnostic'].value_counts()

    counts.plot.pie(autopct='%1.1f%%')
    plt.ylabel('')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'diag-percentage-dist.png'), dpi=300, bbox_inches='tight')
        plt.close()
    if show:
        plt.show()
        plt.close()

def plot_images_per_patient_distribution(df, save=False, show=True):
    patient_counts = df['patient_id'].value_counts()

    distribution = patient_counts.value_counts().sort_index()

    distribution.plot(kind='bar')
    plt.xlabel('Number of images per patient')
    plt.ylabel('Number of patients')

    for i, v in enumerate(distribution.values):
        plt.text(i, v + 2, str(v), ha='center')
        
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'images_per_patient_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    if show:
        plt.show()
        plt.close()
        
def plot_patiens_per_diagnosis(df, save=False, show=True):
    # Get one diagnosis per patient
    patient_diagnosis = df.groupby('patient_id')['diagnostic'].first()

    # Count patients per diagnosis
    diagnosis_counts = patient_diagnosis.value_counts()

    # Plot - capture the axis
    ax = diagnosis_counts.plot(kind='bar')
    plt.xlabel('Diagnosis')
    plt.ylabel('Number of patients')
    plt.xticks(rotation=45)
    
    # Add values on bars - use ax instead of plt
    for i, v in enumerate(diagnosis_counts.values):  # Add .values here
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'patiens_per_diagnosis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    if show:
        plt.show()
        plt.close()
    
if __name__ == "__main__":            
    df = pd.read_csv(METADATA_CSV_PATH)
    #plot_diagnosis_distribution(df, save=True)
    #plot_diagnosis_percentage_distribution(df, save=True)
    plot_images_per_patient_distribution(df, save=True, show=False)