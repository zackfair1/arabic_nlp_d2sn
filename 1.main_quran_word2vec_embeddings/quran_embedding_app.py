import pandas as pd
from gensim.models import Word2Vec
import plotly.express as px
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
import tqdm
tqdm.tqdm.pandas()
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# For dash web app
from dash import Dash, dcc, html, Input, Output

class pca_umap():
    """This class serves for visualizing PCA & UMAP based on search text. Note that it may not work with other types of dataframes or may take quite the time to load larger datasets"""

    def __init__(self):
        # Model
        self.model = Word2Vec.load('https://github.com/zackfair1/reddit_files/blob/master/quran_w2v.model?raw=true')
        # PCA
        self.pca_df = pd.read_csv('https://raw.githubusercontent.com/zackfair1/reddit_files/master/pca_df.csv')
        # UMAP
        self.umap_df = pd.read_csv('https://raw.githubusercontent.com/zackfair1/reddit_files/master/umap_df.csv', index_col=0)
        
    def find_most_similar(self, word): # <--- Most similar words
        try:
            list_words = []
            if type(word) == list:
                for w in word:
                    list_words.append(list(np.stack(self.model.wv.most_similar(w), axis=1)[0]) + [w])
                return self.flatten(list_words)
            else:
                word = [word]
                return list(np.stack(self.model.wv.most_similar(word), axis=1)[0]) + word
        except:
            return
    
    def map_ref(self, value, dictionary): # <--- Map the dictionary for the right keys...
        for k, v in dictionary.items():
            if value in v:
                return k
            
    def df_most_similar_(self, df, word):
        try:
            if type(word) == list:
                lookup = []
                reference = {}
                for w in word:
                    for sim in self.find_most_similar(w):
                        lookup.append(sim)
                        reference[sim] = w
                df = df[df.word.isin(lookup)]
                reference = {v:[i for i in reference.keys() if reference[i] == v ] for k,v in reference.items()}
                df['ref'] = df['word'].progress_map(lambda x : self.map_ref(x, reference))
                return df
            else:
                df = df[df.word.isin(self.find_most_similar(word))]
                df['ref'] = word
                return df
        except Exception as e:
            return f'Try some other words. Error: {e}'
        
    def plot_pca_similar(self, words): # <--- PCA Similar words to particular word(s) plot
        df = self.df_most_similar_(self.pca_df.reset_index(), words)
        fig_pca = px.scatter_3d(df, x='dim1', y='dim2', z='dim3', color='ref', text='word')
        fig_pca.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        # fig_pca.show()
        return fig_pca
    
    def plot_umap_similar(self, words):
        df = self.df_most_similar_(self.umap_df, words)
        fig_umap = px.scatter_3d(df, x='dim1',y='dim2',z='dim3',text='word',color='ref')
        fig_umap.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        # fig_umap.show()
        return fig_umap

w2v = pca_umap()

app = Dash(__name__)
app.layout = html.Div([
    html.H1('Quran Word Embeddings', style={'flex-basis':'100%','width':'51%','text-align':'center','text-transform':'uppercase','font-size':'2em','font-weight':'bold'}), # Title

    html.Div([                  # <---- PCA
        html.H3('PCA', style={'text-align':'center'}),
        html.Div([              # <---- dropdown 2
        dcc.Dropdown(options=w2v.pca_df.word.unique(), value=['الله'], id='demo-dropdown-2', multi=True),
        html.Div(id='dd-output-container-2', style={'display':'none'}),
        dcc.Graph(id='figure_2'),
    ], style={'height':'50vh', 'margin':'2vw'}),
    ], style={'width':'75%', 'box-shadow':'-1px 3px 18px -6px rgba(0,0,0,0.15)','border-radius':'1%', 'margin':'2vh 0'}),
    
    html.Div([                  # <---- UMAP
        html.H3('UMAP', style={'text-align':'center'}),
        html.Div([              # <---- dropdown 1
        dcc.Dropdown(options=w2v.umap_df.word.unique(), value=['الله'], id='demo-dropdown', multi=True),
        html.Div(id='dd-output-container', style={'display':'none'}),
        dcc.Graph(id='figure_1'),
    ], style={'height':'50vh', 'margin':'2vw'}),
    ], style={'width':'75%', 'box-shadow':'-1px 3px 18px -6px rgba(0,0,0,0.15)','border-radius':'1%', 'margin':'2vh 0'}),
    
    html.Div([                  # <---- Tensorboard
        html.H3('Tensorboard Embedding Projector', style={'text-align':'center'}),
        html.Iframe(src='https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/zackfair1/reddit_files/master/q_config.json', style={'width':'100%', 'box-shadow':'-1px 3px 18px -6px rgba(0,0,0,0.15)','border-radius':'1%', 'height':'100%','margin':'2vh 0'})
    ], style={'width':'95%', 'box-shadow':'-1px 3px 18px -6px rgba(0,0,0,0.15)','border-radius':'1%', 'height':'100vh'}),
    
    
], style={'width':'95vw', 'display':'flex','justify-content':'space-around','flex-wrap':'wrap', 'font-family':'Arial, sans-serif','margin':'25px auto', 'margin':'0 0 2vh 0'})

@app.callback(
    Output('dd-output-container', 'children'),
    Output('figure_1', 'figure'),
    [Input('demo-dropdown', 'value')]
)
def update_output(value):
    return value, w2v.plot_umap_similar(value) #, f'You have selected {value}'

@app.callback(
    Output('dd-output-container-2', 'children'),
    Output('figure_2', 'figure'),
    [Input('demo-dropdown-2', 'value')]
)
def update_output(value):
    return value,w2v.plot_pca_similar(value) #, f'You have selected {value}'

if __name__ == '__main__':
    app.run_server(debug=False)