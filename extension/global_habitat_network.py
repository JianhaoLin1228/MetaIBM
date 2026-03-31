
# -*- coding: utf-8 -*-
import numpy as np
import random


def _init_global_habitat_network_fields(self):
    self.global_habitat_id_ls = []                           # element: (patch_id, habitat_id) 
    self.global_habitat_id_2_index_dir = {}                  # {(patch_id, habitat_id): global_habitat_index}; row/col index for global_habitat_distance_matrix
    self.global_habitat_distance_matrix = np.matrix([])      # Full-connected Network: element: distance between pairwise habitats at metacommunity scales


def global_habitat_id_idx_registry(self, patch_object):
    patch_id = patch_object.name
    for h_id, h_object in patch_object.set.items():
        self.global_habitat_id_ls.append((patch_id, h_id))
        self.global_habitat_id_2_index_dir[(patch_id, h_id)] = len(self.global_habitat_id_ls) - 1
    return 0


def update_global_habitat_distance_matrix(self):
    hab_num = len(self.global_habitat_id_ls)
    if hab_num == 0:
        dist_matrix = np.matrix([])
    else:
        dist_matrix = np.zeros((hab_num, hab_num), dtype=float)

    for i, (patch_i_id, h_i_id) in enumerate(self.global_habitat_id_ls):
        h_i_object = self.set[patch_i_id].set[h_i_id]
        xi, yi = h_i_object.location
        for j, (patch_j_id, h_j_id) in enumerate(self.global_habitat_id_ls):
            h_j_object = self.set[patch_j_id].set[h_j_id]
            xj, yj = h_j_object.location
            dist_matrix[i, j] = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    self.global_habitat_distance_matrix = np.matrix(dist_matrix)
    return self.global_habitat_distance_matrix


def incremental_update_global_habitat_distance_matrix(self):
    new_hab_num = len(self.global_habitat_id_ls)
    old_hab_num = self.global_habitat_distance_matrix.shape[0]

    if self.global_habitat_distance_matrix.size == 0:
        return self.update_global_habitat_distance_matrix()

    dist_matrix = np.zeros((new_hab_num, new_hab_num), dtype=float)
    dist_matrix[0:old_hab_num, 0:old_hab_num] = self.global_habitat_distance_matrix

    for i in range(old_hab_num, new_hab_num):
        patch_i_id, h_i_id = self.global_habitat_id_ls[i]
        h_i_object = self.set[patch_i_id].set[h_i_id]
        xi, yi = h_i_object.location
        for j in range(0, new_hab_num):
            patch_j_id, h_j_id = self.global_habitat_id_ls[j]
            h_j_object = self.set[patch_j_id].set[h_j_id]
            xj, yj = h_j_object.location
            dij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            dist_matrix[i, j] = dij
            dist_matrix[j, i] = dij

    self.global_habitat_distance_matrix = np.matrix(dist_matrix)
    return self.global_habitat_distance_matrix

def get_global_habitat_network_dormancy_pool_num_matrix(self):
    ''' Return a diagonal matrix Dormancy for the global habitat network, Dormancy[i, i] = number of dormancy propagules in habitat i '''
    hab_num = len(self.global_habitat_id_ls)
    global_hab_net_dormancy_pool_num_matrix = np.matrix(np.zeros((hab_num, hab_num), dtype=float))
    for idx, (patch_id, h_id) in enumerate(self.global_habitat_id_ls):
        global_hab_net_dormancy_pool_num_matrix[idx, idx] = len(self.set[patch_id].set[h_id].dormancy_pool)
    return global_hab_net_dormancy_pool_num_matrix

def get_global_habitat_network_offspring_pool_num_matrix(self):
    ''' Return a diagonal matrix O for the global habitat network, O[i, i] = number of offspring in habitat i '''
    hab_num = len(self.global_habitat_id_ls)
    global_hab_net_offspring_pool_num_matrix = np.matrix(np.zeros((hab_num, hab_num), dtype=float))

    for idx, (patch_id, h_id) in enumerate(self.global_habitat_id_ls):
        global_hab_net_offspring_pool_num_matrix[idx, idx] = len(self.set[patch_id].set[h_id].offspring_pool)
    return global_hab_net_offspring_pool_num_matrix

def get_global_habitat_network_offspring_marker_pool_num_matrix(self):
    ''' Return a diagonal matrix O for the global habitat network, O[i, i] = number of offspring markers in habitat i '''
    hab_num = len(self.global_habitat_id_ls)
    global_hab_net_offspring_marker_num_matrix = np.matrix(np.zeros((hab_num, hab_num), dtype=float))

    for idx, (patch_id, h_id) in enumerate(self.global_habitat_id_ls):
        global_hab_net_offspring_marker_num_matrix[idx, idx] = len(self.set[patch_id].set[h_id].offspring_marker_pool)
    return global_hab_net_offspring_marker_num_matrix
    
def get_global_habitat_network_empty_sites_num_matrix(self):
    ''' Return a diagonal matrix Empty for the global habitat network, Empty[i, i] = number of empty microsites in habitat i '''
    hab_num = len(self.global_habitat_id_ls)
    global_hab_net_empty_sites_num_matrix = np.matrix(np.zeros((hab_num, hab_num), dtype=float))
    for idx, (patch_id, h_id) in enumerate(self.global_habitat_id_ls):
        global_hab_net_empty_sites_num_matrix[idx, idx] = len(self.set[patch_id].set[h_id].empty_site_pos_ls)
    return global_hab_net_empty_sites_num_matrix

def calculate_global_habitat_network_dispersal_kernel_strength_matrix(self, method='uniform', **kwargs):
    ''' distance matrix to dispersal kernel matrix '''
    hab_num = len(self.global_habitat_id_ls)
    unnormalized_D = np.zeros((hab_num, hab_num), dtype=float)
    for i, (patch_i_id, h_i_id) in enumerate(self.global_habitat_id_ls):
        for j, (patch_j_id, h_j_id) in enumerate(self.global_habitat_id_ls):
            if i == j: unnormalized_D[i, j] = 0.0
            elif patch_i_id == patch_j_id: unnormalized_D[i, j] = 0.0
            else:
                d_ij = self.global_habitat_distance_matrix[i, j]
                unnormalized_D[i, j] = self.dispersal_kernel_strength(d_ij, method=method, **kwargs)
    return np.matrix(unnormalized_D)

def normalized_calculate_global_habitat_network_dispersal_kernel_strength_matrix(self, method='uniform', **kwargs):
    D = self.calculate_global_habitat_network_dispersal_kernel_strength_matrix(method=method, **kwargs)
    normalized_D = D/D.sum(axis=1)
    return normalized_D

def global_habitat_dispersal_among_rate_matrix(self, total_disp_among_rate, method='uniform', **kwargs):
    D_normalized = self.normalized_calculate_global_habitat_network_dispersal_kernel_strength_matrix(method=method, **kwargs)
    M = total_disp_among_rate * D_normalized
    M[np.diag_indices_from(M)] = 1 - total_disp_among_rate  # elements in diagonal_line = 1-m
    return M
    
def get_global_habitat_network_emigrants_matrix(self, total_disp_among_rate, method='uniform', **kwargs):
    ''' the element em_i_j is the expectation of emigrants_num disperse from habitat_i to habitat_j in the global_habitat_network '''
    return (self.get_global_habitat_network_offspring_pool_num_matrix() + self.get_global_habitat_network_dormancy_pool_num_matrix()) * self.global_habitat_dispersal_among_rate_matrix(total_disp_among_rate, method=method, **kwargs)

def get_global_habitat_network_immigrants_matrix(self, total_disp_among_rate, method='uniform', **kwargs):
    ''' the element im_i_j is the allocated number of empty microsites in habitat_j for propagules from habitat_i '''
    emigrants_matrix = self.get_global_habitat_network_emigrants_matrix(total_disp_among_rate, method=method, **kwargs)
    empty_sites_num_matrix = self.get_global_habitat_network_empty_sites_num_matrix()
    return (emigrants_matrix / emigrants_matrix.sum(axis=0)) * empty_sites_num_matrix
    
def get_global_habitat_network_disp_among_num_matrix(self, total_disp_among_rate, method='uniform', **kwargs):
    ''' the element da_i_j is the realized number of propagules disperse from habitat_i to habitat_j '''
    global_hab_net_emigrants_matrix = self.get_global_habitat_network_emigrants_matrix(total_disp_among_rate, method=method, **kwargs)
    global_hab_net_immigrants_matrix = self.get_global_habitat_network_immigrants_matrix(total_disp_among_rate, method=method, **kwargs)
    return self.matrix_around(np.minimum(global_hab_net_emigrants_matrix, global_hab_net_immigrants_matrix))

def dispersal_among_patches_in_global_habitat_network_from_offspring_pool_to_immigrant_pool(self, total_disp_among_rate, method='uniform', **kwargs):
    ''' dispersal from habitat_i offspring_pool to habitat_j immigrant_pool in the global habitat network '''
    if self.patch_num < 2:
        log_info = '[Dispersal among patches in global habitat network] in %s: patch_num < 2, there are 0 individuals disperse into immigrant_pool among habitats \n' % self.metacommunity_name
        return log_info
        
    global_hab_net_offspring_dispersal_matrix = self.matrix_around(self.get_global_habitat_network_offspring_pool_num_matrix() * self.global_habitat_dispersal_among_rate_matrix(total_disp_among_rate, method=method, **kwargs))
    counter = 0
    for j, (patch_j_id, h_j_id) in enumerate(self.global_habitat_id_ls):
        h_j_object = self.set[patch_j_id].set[h_j_id]
        migrants_to_j_indi_object_ls = []

        for i, (patch_i_id, h_i_id) in enumerate(self.global_habitat_id_ls):
            if i == j or patch_i_id == patch_j_id:
                continue
                
            migrants_i_j_num = int(global_hab_net_offspring_dispersal_matrix[i, j])
            h_i_object = self.set[patch_i_id].set[h_i_id]
            migrants_to_j_indi_object_ls += random.sample(h_i_object.offspring_pool, migrants_i_j_num)

        random.shuffle(migrants_to_j_indi_object_ls)
        h_j_object.immigrant_pool += migrants_to_j_indi_object_ls
        counter += len(migrants_to_j_indi_object_ls)

    log_info = '[Dispersal among patches in global habitat network] in %s: there are %d individuals disperse into immigrant_pool among patches in global-habitat-network; there are %d individuals in the immigrant pools in the metacommunity \n' % (self.metacommunity_name, counter, self.meta_immigrant_pool_individual_num())
    return log_info

def dispersal_among_patches_in_global_habitat_network_from_offspring_marker_pool_to_immigrant_marker_pool(self, total_disp_among_rate, method='uniform', **kwargs):
    ''' dispersal from habitat_i offspring_marker_pool to habitat_j immigrant_marker_pool in the global habitat network '''
    if self.patch_num < 2:
        log_info = '[Dispersal among patches in global habitat network] in %s: patch_num < 2, there are 0 offspring markers disperse into immigrant_marker_pool among habitats \n' % self.metacommunity_name
        return log_info
    global_hab_net_offspring_marker_dispersal_matrix = self.matrix_around(self.get_global_habitat_network_offspring_marker_pool_num_matrix() * self.global_habitat_dispersal_among_rate_matrix(total_disp_among_rate, method=method, **kwargs))
    counter = 0
    for j, (patch_j_id, h_j_id) in enumerate(self.global_habitat_id_ls):
        h_j_object = self.set[patch_j_id].set[h_j_id]
        migrants_to_j_offspring_marker_ls = []

        for i, (patch_i_id, h_i_id) in enumerate(self.global_habitat_id_ls):
            if i == j or patch_i_id == patch_j_id:
                continue

            migrants_i_j_num = int(global_hab_net_offspring_marker_dispersal_matrix[i, j])
            h_i_object = self.set[patch_i_id].set[h_i_id]
            migrants_to_j_offspring_marker_ls += random.sample(h_i_object.offspring_marker_pool, migrants_i_j_num)

        random.shuffle(migrants_to_j_offspring_marker_ls)
        h_j_object.immigrant_marker_pool += migrants_to_j_offspring_marker_ls
        counter += len(migrants_to_j_offspring_marker_ls)

    log_info = '[Dispersal among patches in global habitat network] in %s: there are %d offspring markers disperse into immigrant_marker_pool among patches in global-habitat-network; there are %d offspring markers in the immigrant_marker_pools in the metacommunity \n' % (self.metacommunity_name, counter, self.meta_immigrant_marker_pool_marker_num())
    return log_info


def install_global_habitat_network_methods(cls):
    old_init = cls.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        _init_global_habitat_network_fields(self)
    cls.__init__ = new_init

    cls.global_habitat_id_idx_registry = global_habitat_id_idx_registry
    cls.update_global_habitat_distance_matrix = update_global_habitat_distance_matrix
    cls.incremental_update_global_habitat_distance_matrix = incremental_update_global_habitat_distance_matrix
    cls.get_global_habitat_network_dormancy_pool_num_matrix = get_global_habitat_network_dormancy_pool_num_matrix
    cls.get_global_habitat_network_offspring_pool_num_matrix = get_global_habitat_network_offspring_pool_num_matrix
    cls.get_global_habitat_network_offspring_marker_pool_num_matrix = get_global_habitat_network_offspring_marker_pool_num_matrix
    cls.get_global_habitat_network_empty_sites_num_matrix = get_global_habitat_network_empty_sites_num_matrix
    cls.calculate_global_habitat_network_dispersal_kernel_strength_matrix = calculate_global_habitat_network_dispersal_kernel_strength_matrix
    cls.normalized_calculate_global_habitat_network_dispersal_kernel_strength_matrix = normalized_calculate_global_habitat_network_dispersal_kernel_strength_matrix
    cls.global_habitat_dispersal_among_rate_matrix = global_habitat_dispersal_among_rate_matrix
    cls.get_global_habitat_network_emigrants_matrix = get_global_habitat_network_emigrants_matrix
    cls.get_global_habitat_network_immigrants_matrix = get_global_habitat_network_immigrants_matrix
    cls.get_global_habitat_network_disp_among_num_matrix = get_global_habitat_network_disp_among_num_matrix
    cls.dispersal_among_patches_in_global_habitat_network_from_offspring_pool_to_immigrant_pool = dispersal_among_patches_in_global_habitat_network_from_offspring_pool_to_immigrant_pool
    cls.dispersal_among_patches_in_global_habitat_network_from_offspring_marker_pool_to_immigrant_marker_pool = dispersal_among_patches_in_global_habitat_network_from_offspring_marker_pool_to_immigrant_marker_pool
    return cls
