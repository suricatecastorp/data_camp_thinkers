import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import nltk
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx
from networkx.algorithms.centrality import degree_centrality,eigenvector_centrality,closeness_centrality
from networkx.algorithms.centrality import betweenness_centrality,subgraph_centrality


class FeatureExtractor(object):
    def __init__(self):
        nltk.download('punkt') # for tokenization
        nltk.download('stopwords')
        self.stpwds = set(nltk.corpus.stopwords.words("english"))
        self.stemmer = nltk.stem.PorterStemmer()
        self.G = nx.Graph()
        self.nodes = None
        self.edges = None

    def fit(self, X_df, y_array):
        max_id = max(X_df['thinker_id_1'].max(),X_df['thinker_id_2'].max())
        self.nodes = np.arange(1,max_id+1)
        self.edges = X_df[y_array['link']==1][['thinker_id_1','thinker_id_2']]
        
        self.G.add_nodes_from(self.nodes)
        self.G.add_edges_from(self.edges.values)
        
        self.graph_features = pd.DataFrame({'thinker_id':self.nodes})
        self.connected_comp = list(nx.connected_components(self.G))
        
        group_id = {}
        group_len = {}
        for think_id in self.nodes:
            for i,group in enumerate(self.connected_comp):
                if think_id in group:
                    group_id[think_id] = i
                    group_len[think_id] = len(group)
                    break

        self.graph_features['connected_comp'] = [group_id[think_id] for think_id in self.nodes]
        self.graph_features['connected_comp_len'] = [group_len[think_id] for think_id in self.nodes]
        
        self.graph_features['degree_centrality'] = degree_centrality(self.G).values()
        self.graph_features['degree_centrality']/=self.graph_features['degree_centrality'].max()
        
        self.graph_features['eigenvector_centrality'] = eigenvector_centrality(self.G).values()
        self.graph_features['eigenvector_centrality']/=self.graph_features['eigenvector_centrality'].max()

        self.graph_features['closeness_centrality'] = closeness_centrality(self.G).values()
        self.graph_features['closeness_centrality']/=self.graph_features['closeness_centrality'].max()

        self.graph_features['betweenness_centrality'] = betweenness_centrality(self.G).values()
        self.graph_features['betweenness_centrality']/=self.graph_features['betweenness_centrality'].max()

        self.graph_features['subgraph_centrality'] = subgraph_centrality(self.G).values()
        self.graph_features['subgraph_centrality']/=self.graph_features['subgraph_centrality'].max()

        self.graph_features['pagerank'] = nx.pagerank(self.G, alpha=0.9).values()
        self.graph_features['pagerank']/=self.graph_features['pagerank'].max()

    def transform(self, X_df):
        left_graph = self.graph_features.rename(columns=lambda name:name+'_1')
        right_graph = self.graph_features.rename(columns=lambda name:name+'_2')
        
        X_encoded = pd.merge(X_df,left_graph,left_on='thinker_id_1',right_on='thinker_id_1',how='left')
        X_encoded = pd.merge(X_encoded,right_graph,left_on='thinker_id_2',right_on='thinker_id_2',how='left')
        
        
        def clean(text):
            text_tokenized = text.lower().split(" ")
            # remove stopwords
            text_tokenized = [token for token in text_tokenized if token not in self.stpwds]
            text_cleaned = [self.stemmer.stem(token) for token in text_tokenized]
            return(text_cleaned)
        def overlap_text(X):
            summary_1 = X['summary_1']
            summary_2 = X['summary_2']
            list_overlap = []
            for i in range(len(summary_1)):
                list_overlap.append(len(set(clean(str(summary_1[i]))).intersection(set(clean(str(summary_2[i]))))))
            return(np.array(list_overlap).reshape(-1, 1))
        overlap_text_transformer = FunctionTransformer(overlap_text, validate=False)

        def to_num(X):
            return(X.apply(func=(lambda x: pd.to_numeric(x, errors='coerce'))).values)
        to_num_transformer = FunctionTransformer(to_num, validate=False)
        numeric_transformer = Pipeline(steps=[('to_num',to_num_transformer),
                                              ('impute', SimpleImputer(strategy='median'))])

        graph_cols = ['connected_comp','connected_comp_len','degree_centrality','eigenvector_centrality',
                      'closeness_centrality','betweenness_centrality','subgraph_centrality','pagerank']
        
        num_cols = ['birth_date_1','birth_date_2','death_date_1','death_date_2',]
        num_cols += [col+'_1' for col in graph_cols]
        num_cols += [col+'_2' for col in graph_cols]

        summary_cols = []
        drop_cols = ['summary_1', 'summary_2','thinker_1','thinker_2','thinker_id_1','thinker_id_2','birth_place_1',
                     'birth_place_2','death_place_2','death_place_2']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('overlap_text', make_pipeline(overlap_text_transformer, SimpleImputer(strategy='median')), summary_cols),
                ('drop cols', 'drop', drop_cols),
                ])

        X_array = preprocessor.fit_transform(X_encoded)
        
        return X_array