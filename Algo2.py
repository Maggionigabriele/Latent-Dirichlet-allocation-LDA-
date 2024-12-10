# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:28:23 2023

@author: maggi
"""

import numpy as np
import numpy.random as rnd
from scipy.special import digamma
from scipy.special import loggamma
from tqdm import tqdm


def init_var_param(C, K):
    print('Initializing variational parameters...')

    # Number of articles, vocabulary size
    D, V = C.shape

    # Topics (initializing LAMBDA for BETA)
    Lambda_0 = rnd.uniform(low=0.01, high=1.0, size=(K,V))

    # Topic Proportions (initializing GAMMA for THETA)
    Gamma_0 = np.full((D,K) , 1.0) #Uniform prior

    # Topic Assignments (initializing PHI for Z)
    # Shape: (N,n_words,C) (Note: n_words is variable)
    Phi_0 = np.full((D,V,K), 1/K)
    for document in range(C.shape[0]):
        for word in range(C.shape[1]):
            if C[document, word] == 0:
                Phi_0[document, word, :] = np.full(K, 0)
            #else:
             #   Phi[document, word, :] = rnd.dirichlet(np.full(K,1))

    return Lambda_0, Gamma_0, Phi_0



def compute_elbo(Lam, Gam, Phi, C, K, eta, alpha):
    elbo = 0

    # Number of articles, vocabulary size
    D, V = C.shape

    # Add expected log joint
    ## First term: \sum_{k=1}^C E[log p(BETA_k)]
    E_log_p_beta = 0
    for k in range(K):
        E_log_p_beta += (eta-1) * np.sum(digamma(Lam[k]) - digamma(np.sum(Lam[k])))

    elbo += E_log_p_beta

    ## Second term: \sum_{i=1}^N E[log p(THETA_i)]
    E_log_p_theta = 0
    for i in range(D):
        E_log_p_theta += (alpha-1) * np.sum(digamma(Gam[i]) - digamma(np.sum(Gam[i])))

    elbo += E_log_p_theta

    ## Third term:
    ## \sum_{i=1}^N \sum_{j=1}^M \sum_{k=1}^C
    ## (E[log p(Z_ij|THETA_i)] + E[log p(X_ij)|BETA,Z_ij)])
    E_log_p_xz = 0
    for document in range(D):
        #article = C[document]

        for word in range(V):
            if C[document,word] != 0:
            ### E[log p(Z_ij|THETA_i)]
                E_log_p_xz += np.sum(Phi[document][word] * (digamma(Gam[document]) - digamma(np.sum(Gam[document]))))
                            #word               Z[document, word, :] * 
            ### E[log p(X_ij|BETA,Z_ij)]
                E_log_p_xz +=  np.sum(Phi[document][word] * (digamma(Lam[:,word]) - digamma(np.sum(Lam, axis=1))))

    elbo += E_log_p_xz

    # Add entropy
    ## Fourth term: -\sum_{k=1}^C E[log q(BETA_k)]
    E_log_q_beta = 0
    for k in range(K):
        E_log_q_beta += -loggamma(np.sum(Lam[k])) + np.sum(loggamma(Lam[k]))
        E_log_q_beta += -np.sum((Lam[k]-1) * (digamma(Lam[k]) - digamma(np.sum(Lam[k]))))

    elbo += E_log_q_beta

    ## Fifth term: -\sum_{i=1}^N E[log q(THETA_i)]
    E_log_q_theta = 0
    for document in range(D):
        E_log_q_theta += -loggamma(np.sum(Gam[document])) + np.sum(loggamma(Gam[document]))
        E_log_q_theta += -np.sum((Gam[document]-1) * (digamma(Gam[document]) - digamma(np.sum(Gam[document]))))

    elbo += E_log_q_theta

    ## Sixth term: -\sum_{i=1}^N \sum_{j=1}^M (E[log q(Z_ij)])
    E_log_q_z = 0
    for document in range(D):
        #article = C[document]

        for word in range(V):
            if C[document,word] != 0:
                E_log_q_z += - np.sum(Phi[document][word] * np.log(Phi[document][word]))


    elbo += E_log_q_z

    print('ELBO: {}'.format(elbo))

    return elbo




def run_cavi2(LAMBDA, GAMMA, PHI, C, K = 8, max_iter=10, ALPHA = 0.5, ETA = 0.5):
    # Unpack initial variational parameters
    LAMBDA_t = np.copy(LAMBDA) # Shape: (K,V)
    GAMMA_t = np.copy(GAMMA) # Shape: (D,K)
    PHI_t = np.copy(PHI) # Shape: (D,V,K)

    # Number of articles, vocabulary size
    D, V = C.shape

    elbos = []

    print('Running CAVI for LDA (K: {}, Iter: {})...'.format(K, max_iter))
    for t in range(max_iter):
        print('Iteration {}'.format(t+1))
        print('Updating PHI and GAMMA')

        # For each document
        i = 0
        while True:
            i += 1
            print ('Iteration PHI and GAMMA{}'.format(i))
            GAMMA_old = np.copy(GAMMA_t)
            for document in tqdm(range(D)):
                article = C[document]
    
                # Fetch for PHI_ij update
                GAMMA_i_t = np.copy(GAMMA_t[document]) # C-vector
                
    
                # Iterate through each word with non-zero count on document
                for word in range(V):
                    if C[document,word] != 0:
                        log_PHI_ij = np.zeros((K,))
    
                        for k in range(K):
                            # Fetch for PHI_ij update
                            LAMBDA_k_t = np.copy(LAMBDA_t[k]) # V-vector
        
                            exponent = digamma(GAMMA_i_t[k]) - digamma(np.sum(GAMMA_i_t))
                            exponent += digamma(LAMBDA_k_t[word]) - digamma(np.sum(LAMBDA_k_t))
                            log_PHI_ij[k] = np.exp(exponent)
    
                    # Normalize using log-sum-exp trick
                        #print(log_PHI_ij)
                        PHI_ij = log_PHI_ij / (np.sum(log_PHI_ij))
                        try:
                            assert(np.abs(np.sum(PHI_ij) - 1) < 1e-6)
                        except:
                            raise AssertionError('phi_ij: {}, sum: {} , gamma: {}, lambda: {}, exponent:{}'.format(PHI_ij, np.sum(PHI_ij),GAMMA_i_t, LAMBDA_k_t , exponent))
    
                    #try:
                     #   assert(np.abs(np.sum(PHI_ij) - 1) < 1e-6)
                    #except:
                     #   raise AssertionError('phi_ij: {}, Sum: {}'.format(PHI_ij, np.sum(PHI_ij)))
    
                        PHI_t[document][word] = PHI_ij
                        
                # Check if number of updates match with number of words
    
                # Update GAMMA_i
                GAMMA_i_t = np.zeros((K,)) 
    
                for k in range(K):
                    GAMMA_i_t[k] = np.sum(PHI_t[document,:,k]) + ALPHA
                #print (GAMMA_i_t, document)
                GAMMA_t[document] = GAMMA_i_t
                #print (GAMMA_t[document])
                #print (GAMMA_i_t)
            #if not predict_flag:
                # For each topic
            #print (GAMMA_t)
            value = np.mean(np.square(GAMMA_t - GAMMA_old))
            print ()
            if value < 0.001:
                break
        print('Mean_differences{}'.format(value))

        for k in tqdm(range(K)):
            LAMBDA_k_t = np.zeros((V,)) + ETA

            # For each document
            for doc in range(D):
                article = C[doc]

                for word in range(V):
                    if article[word] != 0:
                        LAMBDA_k_t[word] += article[word] * PHI_t[doc][word][k]


            LAMBDA_t[k] = LAMBDA_k_t

        # Compute ELBO
        #print (GAMMA_t)
        elbo = compute_elbo(LAMBDA_t, GAMMA_t, PHI_t, C, K, ETA, ALPHA)
        elbos.append(elbo)

    LAMBDA_final = np.copy(LAMBDA_t)
    GAMMA_final = np.copy(GAMMA_t)
    PHI_final = np.copy(PHI_t)

    return LAMBDA_final, GAMMA_final, PHI_final, elbos

############################
############################
############################


import pickle as pkl

jmlr_papers = pkl.load(open("jmlr.pkl","rb"))

bayesian_jmlr_papers = []

for paper in jmlr_papers:
    bayesian_keywords = ['graph', 'dirichlet']
    if any([kwd in paper["abstract"] for kwd in bayesian_keywords]):
        bayesian_jmlr_papers.append(paper)
        
print("There are", str(len(bayesian_jmlr_papers))+" Bayesian papers out of", str(len(jmlr_papers)))



from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features = 1000, stop_words='english')
X = vectorizer.fit_transform([paper["abstract"] for paper in jmlr_papers])
print(vectorizer.get_feature_names_out()) # Top-1000 words
C = X.toarray() # Count matrix
# Removing documents with 0 words
idx = np.where(np.sum(C, axis = 1)==0)
C = np.delete(C, idx, axis = 0)


Words = vectorizer.get_feature_names_out()

Lambda_0_2, Gamma_0_2, Phi_0_2 = init_var_param(C = C, K = 8)

LAMBDA_final_2, GAMMA_final_2, PHI_final_2, elbos_2 = run_cavi2(LAMBDA = Lambda_0_2, GAMMA = Gamma_0_2, PHI = Phi_0_2, C = C, K = 8, max_iter=100, ALPHA = 0.5, ETA = 0.5)


Words[np.argpartition(-LAMBDA_final[1,:], 10)[0:10]]