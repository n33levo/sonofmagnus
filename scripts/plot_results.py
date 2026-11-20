"""Plot the published local estimates, conditional on their fixed engine anchors."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'results'


def main():
    rows=json.loads((ROOT/'local-pool.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.spines.left':False,'figure.facecolor':'#fafbfc','axes.facecolor':'#fafbfc',
        'text.color':'#162438','xtick.color':'#526173','ytick.color':'#526173','svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(9,3.1))
    for y,(name,row) in enumerate(reversed(list(rows.items()))):
        value=row['local_elo'];low,high=row['elo_opening_bootstrap_95']
        ax.errorbar(value,y,xerr=[[value-low],[high-value]],fmt='o',color='#087e8b',capsize=5,markersize=9,lw=2)
        ax.annotate(f'{value:.0f}  [{low:.0f}, {high:.0f}]',(value,y),xytext=(0,15),textcoords='offset points',ha='center',weight='bold')
    ax.set(yticks=[0,1],yticklabels=['6M CNN','185M policy'],ylim=(-.4,1.55),xlim=(1900,3100),xlabel='CCRL-referenced local Elo')
    ax.set_title('Searchless inference · 400 games per model',loc='left',fontsize=17,weight='bold',pad=18)
    ax.grid(axis='x',alpha=.15)
    fig.text(.15,-.01,'95% opening-cluster bootstrap intervals, conditional on fixed anchors.\nAnchor hardware/clock differences are excluded; these are not official Lichess or FIDE ratings.',fontsize=9,color='#526173')
    fig.tight_layout(rect=(0,.08,1,1));directory=ROOT/'figures';directory.mkdir(exist_ok=True)
    fig.savefig(directory/'local-pool.png',bbox_inches='tight',dpi=160,metadata={'Creator':'sonofmagnus'});plt.close(fig)


if __name__=='__main__':main()
