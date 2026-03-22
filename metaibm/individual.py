# -*- coding: utf-8 -*-
"""
Split from metacommunity_IBM.py.
Compatibility-preserving class module.
"""

import numpy as np
import random


class individual():
    def __init__(self, species_id, traits_num, pheno_names_ls, gender='female', genotype_set=None, phenotype_set=None):
        self.species_id = species_id
        self.gender = gender
        self.traits_num = traits_num
        self.pheno_names_ls = pheno_names_ls
        self.genotype_set = genotype_set
        self.phenotype_set = phenotype_set
        self.age = 0
        
    def random_init_indi(self, mean_pheno_val_ls, pheno_var_ls, geno_len_ls):
        '''
        pheno_names is a tuple of the pheno_names (string) i.e., ('phenotye_1', 'phenotype_2',...,'phenotye_x') and the len(pheno_names) is equal to traits_num.
        mean_pheno_val (tuple) is the mean values (float) of the phenotypes of a species population which fit a gaussian distribution, i.e., (val1, val2,...,valx).
        pheno_var (tuple) is the variation (float) of the phenotypes of a species population which fit a gaussian distribution.
        geno_len (tuple) is the len of each genotype in the genotype_set, the genotype in which controls each phenotype of each trait.
        '''
        genotype_set = {}
        phenotype_set = {}
        
        for i in range(self.traits_num):
            name = self.pheno_names_ls[i]
            mean = mean_pheno_val_ls[i]
            var = pheno_var_ls[i]
            geno_len = geno_len_ls[i]
            
            #random_index = random.sample(range(0,geno_len*2),int(mean*geno_len*2))
            #genotype = np.array([1 if i in random_index else 0 for i in range(geno_len*2)])
            #bi_genotype = [genotype[0:geno_len], genotype[geno_len:geno_len*2]]
            
            random_index_1 = random.sample(range(0,geno_len),int(mean*geno_len))
            random_index_2 = random.sample(range(0,geno_len),int(mean*geno_len))
            genotype_1 = np.array([1 if i in random_index_1 else 0 for i in range(geno_len)])
            genotype_2 = np.array([1 if i in random_index_2 else 0 for i in range(geno_len)])
            
            bi_genotype = [genotype_1, genotype_2]
            phenotype = mean + random.gauss(0, var)
            
            genotype_set[name] = bi_genotype
            phenotype_set[name] = phenotype
        self.genotype_set = genotype_set
        self.phenotype_set = phenotype_set
        return 0
    
    def __str__(self):
        species_id_str = 'speceis_id=%s'%self.species_id
        gender_str = 'gender=%s'%self.gender
        traits_num_str = 'traits_num=%d'%self.traits_num
        genotype_set_str = 'genetype_set=%s'%str(self.genotype_set)
        phenotype_set_str = 'phenotype_set=%s'%str(self.phenotype_set)
        
        strings = species_id_str+'\n'+ gender_str+'\n'+traits_num_str+'\n'+genotype_set_str+'\n'+phenotype_set_str
        return strings
    
    def get_indi_phenotype_ls(self):
        indi_phenotype_ls = []
        for pheno_name in self.pheno_names_ls:
            phenotype = self.phenotype_set[pheno_name]
            indi_phenotype_ls.append(phenotype)
        return indi_phenotype_ls
    
    def mutation(self, rate, pheno_var_ls):
        for i in range(self.traits_num):
            mutation_counter = 0
            pheno_name = self.pheno_names_ls[i]
            var = pheno_var_ls[i]
            genotype1 = self.genotype_set[pheno_name][0]
            genotype2 = self.genotype_set[pheno_name][1]
            for index in range(len(genotype1)):
                if rate > np.random.uniform(0,1,1)[0]:
                    mutation_counter += 1
                    if genotype1[index] == 0: self.genotype_set[pheno_name][0][index]=1
                    elif genotype1[index] == 1: self.genotype_set[pheno_name][0][index]=0
                    
            for index in range(len(genotype2)):
                if rate > np.random.uniform(0,1,1)[0]:
                    mutation_counter += 1
                    if genotype2[index] == 0: self.genotype_set[pheno_name][1][index]=1
                    elif genotype2[index] == 1: self.genotype_set[pheno_name][1][index]=0
            if mutation_counter >=1: 
                phenotype = np.mean(self.genotype_set[pheno_name]) + random.gauss(0, var)
                self.phenotype_set[pheno_name] = phenotype
        return 0
