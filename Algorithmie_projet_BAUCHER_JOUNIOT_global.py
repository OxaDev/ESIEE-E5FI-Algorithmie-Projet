import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import argparse

###################################################
############# Robot sur Grille ####################

def RobotGrille_approche_gloutonne(mat_N, mat_NE, mat_E):
    x,y = (0,0)
    m = len(mat_N)
    n = len(mat_N[0])
    #print("m : "+ str(m) + " - n : " + str(n))
    chemin = [(x,y)]
    cout_chemin = 0
    
    while((x,y) != (m-1,n-1) ):
        
        if(x == m-1 or y == n-1):
            if(x == m-1) : cost_N = float('inf')
            else : cost_N = mat_N[x][y]
            if(y == n-1) : cost_E = float('inf')
            else : cost_E = mat_E[x][y]
            cost_NE = float('inf')
        else: 
            cost_N = mat_N[x][y]
            cost_E = mat_E[x][y]
            cost_NE = mat_NE[x][y] 
        
        #print("(x,y)="+str((x,y)) + " -N:"+str(cost_N) + " -E:"+str(cost_E) + " -NE:"+ str(cost_NE))
        min_dir = min(cost_N, cost_NE, cost_E)
        
        if(min_dir == cost_N):
            x += 1
            chemin.append((x,y))
            cout_chemin += min_dir
            continue
        
        if(min_dir == cost_E):
            y += 1
            chemin.append((x,y))
            cout_chemin += min_dir
            continue
        
        if(min_dir == cost_NE):
            x += 1
            y += 1
            chemin.append((x,y))
            cout_chemin += min_dir
            continue
        
    return cout_chemin, chemin

def RobotGrille_approche_optimale(mat_N, mat_NE, mat_E) :
    m = len(mat_N)
    n = len(mat_N[0])
    
    M = [ [float("inf") for c in range(n)] for l in range(m)]

    M[0][0] = 0
    for x in range(1,m) : M[x][0] = M[x-1][0] + mat_N[x-1][0]
    for y in range(1,n) : M[0][y] = M[0][y-1] + mat_E[0][y-1]

    for x in range(1,m) :
        for y in range(1,n) :
                
            if(x-1 == m-1 and y-1 == n-1):NE = float("inf")
            else:
                NE = mat_NE[x-1][y-1]
                
                if(y-1 == n-1): E = float('inf')
                else: E = mat_E[x][y-1]

                if(x-1 == m-1): N = float('inf')
                else: N = mat_N[x-1][y]
            
            
            M[x][y] = min(
            N + M[x-1][y],
            E + M[x][y-1],
            NE + M[x-1][y-1]
            )
    return M

def RobotGrille_afficher(M) :
    m,n = len(M),len(M[0])
    for x in range(m-1,-1,-1) :
        for y in range(n) :
            print(str(M[x][y]) + "\t", end="")
        print("")
        
def RobotGrille_accm(x,y,M,mat_N,mat_NE, mat_E) :
    m = len(mat_N)
    n = len(mat_N[0])
    
    if (x,y) == (0,0) :
        print( str((x,y)), end=" " ) 
        return
    if y == 0 :
        accm(x-1,0,M,mat_N,mat_NE, mat_E) 
        c = mat_N[x-1][0]
        print("--" + str(c) + "--> " + str((x,y)), end=" " )
        return
    if x == 0 :
        accm(0,y-1,M,mat_N,mat_NE, mat_E)
        c = mat_E[0][y-1]
        print( "--" + str(c) + "--> " + str((x,y)), end=" " )
        return

    cminN = mat_N[x-1][y] + M[x-1][y]
    cminE = mat_E[x][y-1] + M[x][y-1]
    cminNE = mat_NE[x-1][y-1] + M[x-1][y-1] 
    if cminN <= min(cminE,cminNE) : 
        accm(x-1,y,M,mat_N,mat_NE, mat_E) 
        print("--" + str(mat_N[x-1][y]) + "--> " + str((x,y)), end=" " )
        return
    elif cminE <= cminNE : 
        accm(x,y-1,M,mat_N,mat_NE, mat_E) 
        print("--" + str(mat_E[x][y-1]) + "--> " + str((x,y)), end=" " )
        return
    else :
        accm(x-1,y-1,M,mat_N,mat_NE, mat_E) 
        print( "--" + str(mat_NE[x-1][y-1]) + "--> " + str((x,y)), end=" " )
        return

###################################################
############# Sac valeur Max ######################

def SacMax_arrangement_glouton(V, E, C):
    V_sort = V
    E_sort = E
    C_temp = C
    
    ## Sort V tab and E tab
    for i in range(len(V_sort)):
        if(i+1 == len(V_sort)):
            break
            
        for j in range(i+1,len(V_sort)):
            if(V_sort[j] > V_sort[i]):
                temp = V_sort[i]
                V_sort[i] = V_sort[j]
                V_sort[j] = temp
                
                temp = E_sort[i]
                E_sort[i] = E_sort[j]
                E_sort[j] = temp
    
    ## Get glouton tab in getting the biggest values of V tab
    glouton_tab = []
    for i in range(len(V_sort)):
        if(C_temp - E_sort[i] >= 0):
            glouton_tab.append(V_sort[i])
            C_temp -= E_sort[i]
    
    return glouton_tab

def SacMax_arrangement_optimal(V,E,C): #CalculerM() function
    n = len(V)
    M = [ [c for c in range(0,C+1)] for k in range(0,n+1)]
    for c in range(C+1) : M[0][c] = 0 # base de la récurrence
    # cas général
    for k in range(1,n+1) :
        for c in range(C+1) :
            if E[k-1] <= c : M[k][c] = max(V[k-1] + M[k-1][c-E[k-1]], M[k-1][c])
            else : # le k-ème objet est trop encombrant pour le sac de contenance c
                M[k][c] = M[k-1][c]
    return M

def SacMax_afficherSac(M,V,E,k,c, i = 0) : 
    if k==0 : 
        return
    if M[k][c] == M[k-1][c] : # le k-ème objet n'est pas dans le sac
        afficherSac(M,V,E,k-1,c, i) # l'affichage du sac "k,c" est obtenu en affichant le sac "k-1,c"
    else : 
        afficherSac(M,V,E,k-1,c-E[k-1],1 ) # afficher le sac "k-1,c-e(k-1)"
        if(i == 0):
            print(str(V[k-1]), end="")
        else:
            print(str(V[k-1]) + ", ", end="")

def SacMax_calculValSac(M,V,E,k,c) : 
    if k==0 : 
        return 0
    if M[k][c] == M[k-1][c] : # le k-ème objet n'est pas dans le sac
        maxi = SacMax_calculValSac(M,V,E,k-1,c) # l'affichage du sac "k,c" est obtenu en affichant le sac "k-1,c"
        return maxi
    else : 
        maxi = SacMax_calculValSac(M,V,E,k-1,c-E[k-1]) # afficher le sac "k-1,c-e(k-1)"
        return maxi + V[k-1]

#######################################################
############# Entrepôts gain max ######################

def Entrepots_approche_gloutonne(G):
    S = len(G[0])-1
    return_tab = []
    for i in range(len(G)):
        return_tab.append([])
        return_tab[i].append(0)
        for j in range(S):
            return_tab[i].append(-1)

    temp_s = S
    while(temp_s > 0 ):
        indic_to_put = Entrepots_getMaxGain(G,return_tab)
        for i in range(1,S+1):
            if(return_tab[indic_to_put][i] == -1):
                return_tab[indic_to_put][i] = G[indic_to_put][i]
                break
        temp_s -= 1
    return return_tab

def Entrepots_getMaxGain(G, already_done):
    indic = 0
    max = -1
    indic_max = 0
    for elem in already_done:
        indic_not_done = 0
        for i in range(len(elem)):
            if(elem[i] == -1):
                indic_not_done = i
                break
        
        if( indic_not_done == 0):
            continue
            
        diffGain = G[indic][indic_not_done] - G[indic][indic_not_done-1]
        if(diffGain > max):
            max = diffGain
            indic_max = indic
        indic += 1

    return indic_max

def Entrepots_approche_optimale(G) : 
    n = len(G) ; S = len(G[0]) - 1; 
    M = [[-1 for s in range(S+1)] for k in range(n+1)] # -1, ou une valeur quelconque.
    A = [[0 for s in range(S+1)] for k in range(n+1)] # 0 ou une valeur quelconque.
    for s in range(S+1) : M[0][s] = 0     # m(0,s) = 0 qqsoit s, 0 ≤ s < S+1
    for k in range(1,n+1) : # par tailles k croissantes
        for s in range(0,S+1) : # pour tout stock s 
            # calculer m(k,s) = max_{0 ≤  s' < s+1} ( g(k-1,s') + m(k-1,s-s') )
            for sprime in range(0,s+1) : 
                mks = G[k-1][sprime] + M[k-1][s-sprime]
                if mks > M[k][s] :
                    M[k][s] = mks
                    A[k][s] = sprime
    return M,A

#####################################################
############# Chemin somme max ######################

def CheminMax_approche_gloutonne(T):
    N = len(T)
    cout = T[0][0]
    path = [0]
    j = 0
    for i in range(1,N):
        if(T[i][j+1] > T[i][j]):
            path.append(j+1)
            j=j+1
        else:
            path.append(j)

        cout += T[i][j]
    return path,cout

def CheminMax_approche_optimale(T,i=0, j=0, M=[]):
    if i == 0:
        # init path tab
        M = []
        for x in range(len(T)):
            M.append([])
            for y in range(len(T[x])):
                M[x].append(0)

    if i == len(T)-1:
        M[i][j] = T[i][j]
        return M
    
    M = CheminMax_approche_optimale(T, i+1, j,M)
    M = CheminMax_approche_optimale(T, i+1, j+1,M)
    
    M[i][j] = T[i][j] + max(M[i+1][j],M[i+1][j+1])
    return M  

#####################################################
############# Robot sur Graphe ######################

def RobotGraphe_approche_gloutonne(G):
    indic_arc = 0
    cout_chemin = 0
    arc_list = [0]
    while(indic_arc != len(G)-1):
        min_arc = 999999
        sommet = G[indic_arc]
        for arc in sommet:
            if(arc[1] < min_arc):
                min_arc = arc[1]
                indic_arc = arc[0]
                
        cout_chemin += min_arc
        arc_list.append(indic_arc)
        
    return cout_chemin, arc_list

def RobotGraphe_calculerGrapheSymetrique(G) : # retourne G' en Theta(m+n).
# L'arc (x,y) du graphe G est l'arc (y,x) du graphe symétrique.
    n = len(G)
    # parcourir G : pour tout arc (x,y) de coût cxy, 
    # ajouter à Gprime l'arc (y,x) de même coût cyx 
    Gprime = [[] for y in range(n)] # un tableau de n listes d'arcs vides.
    for x in range(n) : 
        for (y,cxy) in G[x] :
            Gprime[y].append((x,cxy))
    # en Theta(n+m)	
    return Gprime

def RobotGraphe_approche_optimal(G) :
    Gprime = RobotGraphe_calculerGrapheSymetrique(G)
    n = len(Gprime) # nombre de sommets
    M = [float("inf") for y in range(n)]
    A = [0 for y in range(n)]
    # base de la récurrence :
    M[0] = 0 ; A[0] = -1;
    for y in range(1,n) :
        Py = Gprime[y] # liste des arcs vers y (liste des arcs entrant sur y)
        for (x,cxy) in Py :
            mx = M[x] + cxy # coût min 0 -----> x -> y
            if mx < M[y] : 
                M[y] = mx
                A[y] = x
    return M,A

def RobotGraphe_accm(y,A) : # A[n] est de tg A[y] = argmin m(y) = le dernier sommet avant y.
    if y == 0 : 
        print(0, end="") 
        return
    # 1 ≤ y < n
    RobotGraphe_accm(A[y],A) # le ccm 0 ------> x a été affiché
    print ("-->" + str(y), end="") # l ccm 0 --------x->y a été affiché.

#########################################
############# Main ######################

parser = argparse.ArgumentParser("Script comparant les versions locales (gloutonnes) et les versions optimales des fonctions de divers problèmes : Un robot sur une grille, un sac de valeur maximale, etc...\nMerci de préciser la fonction que vous souhaitez tester, ou utiliser --all pour tout executer")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--robot_grille", default=False, action="store_true", help="Lance l'execution de tests sur les fonctions de robots sur la grille")
group.add_argument("--sac_max", default=False, action="store_true", help="Lance l'execution de tests sur les fonctions du sac de valeur maximum")
group.add_argument("--entrepots", default=False, action="store_true", help="Lance l'execution de tests sur les fonctions de gain max sur la répartition en entrepôts")
group.add_argument("--chemin_max", default=False, action="store_true", help="Lance l'execution de tests sur les fonctions de la recherche d'une somme maximale sur un chemin")
group.add_argument("--robot_graphe", default=False, action="store_true", help="Lance l'execution de tests sur les fonctions de la recherche d'un chemin minimim d'un robot sur un graphe")
group.add_argument("--all", default=False, action="store_true", help="Lance l'execution de tout les tests")

args = parser.parse_args()

nb_tests = 750
save_dir = "graphs"
fact_num_bins = 10

if( args.robot_grille or args.all ):
    print('-- Robot sur Grille --')
    list_val_opti = []
    list_val_gloutonne = []
    list_dist_relativ = []

    for x in range(nb_tests):
        n = random.randint(5,20)
        m = random.randint(5,20)
        mat_N = [[random.randint(1,50) for j in range(n)] for i in range(m)]
        mat_E = [[random.randint(1,50) for j in range(n)] for i in range(m)]
        mat_NE = [[random.randint(1,50) for j in range(n)] for i in range(m)]
        
        M = RobotGrille_approche_optimale(mat_N, mat_NE, mat_E)
        cout_glouton, chemin = RobotGrille_approche_gloutonne(mat_N, mat_NE, mat_E)
        cout_optimal = M[m-1][n-1]
        
        list_val_gloutonne.append(cout_glouton)
        list_val_opti.append(cout_optimal)
        list_dist_relativ.append((cout_glouton - cout_optimal)/cout_optimal)

    num_bins = len(list_dist_relativ)//fact_num_bins 
    plt.hist(list_dist_relativ, num_bins, facecolor = 'blue', alpha = 1)
    plt.title("Robot sur Grille : Distance relative entre méthode gloutonne et optimale")
    plt.ylabel("Nombre d'occurences")
    plt.xlabel("Distance relative")
    plt.savefig(join(save_dir,"Robot_sur_grille.png"))

if( args.sac_max or args.all ):
    print('-- Sac de Valeur Max --')
    list_val_opti = []
    list_val_gloutonne = []
    list_dist_relativ = []
    for i in range(nb_tests):
        V = []
        E = []
        C = random.randint(10,2000)
        N = random.randint(1, 200)
        for i in range(N):
            V.append(random.randint(1, 500))
            E.append( random.randint(1, int(C/2) ) ) 

        ## Approche gloutonne
        glouton_tab = SacMax_arrangement_glouton(V,E,C)
        sum_glouton = 0
        for elem in glouton_tab:
            sum_glouton += elem

        ##Approche optimale
        optimal_tab = SacMax_arrangement_optimal(V,E,C)
        valMax = SacMax_calculValSac(optimal_tab,V,E,len(V),C)
        
        list_val_gloutonne.append(sum_glouton)
        list_val_opti.append(valMax)
        list_dist_relativ.append((valMax - sum_glouton)/valMax)

    num_bins = len(list_dist_relativ)//fact_num_bins 
    plt.hist(list_dist_relativ, num_bins, facecolor = 'blue', alpha = 1)
    plt.title("Sac valeur Max : Distance relative entre méthode gloutonne et optimale")
    plt.ylabel("Nombre d'occurences")
    plt.xlabel("Distance relative")
    plt.savefig(join(save_dir,"Sac_valeur_Max.png"))

if( args.entrepots or args.all ):
    print('-- Entrepôts de gain max --')
    optimal_list = []
    glouton_list = []
    dist_list = []

    for x in range(nb_tests):
        S = random.randint(3,10)
        N = random.randint(3,15)
        G = []
        for i in range(N):
            G.append([0])
            for j in range(1,S):
                G[i].append( G[i][j-1] + random.randint(0,5) )

        ### Approche gloutonne ###
        M_glouton = Entrepots_approche_gloutonne(G)

        # Calcul gain maximal glouton #
        max_gain = 0
        for elem in M_glouton:
            for i in range(1,len(elem)):
                if(elem[i] == -1):
                    max_gain += elem[i-1]
                    break
                if(i+1 == len(elem)):
                    max_gain += elem[i]
                    break

        ### Approche optimale ###
        MA = Entrepots_approche_optimale(G)
        A = MA[1]        # A[0:n+1][0:S+1] de terme général a(k,s) = argmax m(k,s)
        n = len(A) - 1 ; S = len(G[0]) - 1
        M = MA[0]


        optimal_list.append(M[n][S])
        glouton_list.append(max_gain)
        dist_list.append((M[n][S] - max_gain) / M[n][S])

    num_bins = len(dist_list)//fact_num_bins 
    plt.hist(dist_list, num_bins, facecolor = 'blue', alpha = 1 )
    plt.title("Entrepôts : Distance relative entre méthode gloutonne et optimale")
    plt.ylabel("Nombre d'occurences")
    plt.xlabel("Distance relative")
    plt.savefig(join(save_dir,"Entrepôts_gain_max.png"))

if( args.chemin_max or args.all ):
    print('-- Chemin de somme max --')
    optimal_list = []
    glouton_list = []
    dist_list = []

    for i in range(nb_tests):
        N = random.randint(5,20)
        temp_Tr = []
        for j in range(N):
            temp_tab = []
            for x in range(j+1):
                elem = random.randint(1,30)
                temp_tab.append(elem)

            temp_Tr.append(temp_tab)

        temp_Mre = CheminMax_approche_optimale(temp_Tr)
        temp_path_glou,temp_cout_glou = CheminMax_approche_gloutonne(temp_Tr)

        optimal_list.append(temp_Mre[0][0])
        glouton_list.append(temp_cout_glou)
        dist_list.append((temp_Mre[0][0] - temp_cout_glou) / temp_Mre[0][0])

    num_bins = len(dist_list)//fact_num_bins
    plt.hist(dist_list, num_bins, facecolor = 'blue', alpha = 1 )
    plt.title("Chemin de somme max : Distance relative entre méthode gloutonne et optimale")
    plt.ylabel("Nombre d'occurences")
    plt.xlabel("Distance relative")
    plt.savefig(join(save_dir,"Chemin_somme_max.png"))

if( args.robot_graphe or args.all ):
    print('-- Robot sur Graphe --')
    list_val_opti = []
    list_val_gloutonne = []
    list_dist_relativ = []

    for x in range(nb_tests):
        n = random.randint(5,20)
        G = []
        for i in range(n+1):
            G.append([])

        for i in range(n):
            nb_arcs = random.randint(1,5)
            arc_in = []
            for j in range(nb_arcs):
                cout = random.randint(1,10)
                arc = random.randint(i+1,n)
                indic = 0
                lim_iter = n
                while(arc in arc_in and indic < lim_iter):
                    arc = random.randint(i+1,n)
                    indic += 1
                if(indic < lim_iter):
                    G[i].append((arc,cout))
                    arc_in.append(arc)

        MA = RobotGraphe_approche_optimal(G)
        M,A = MA
        n = len(G)
        cout_optimal = M[n-1]
        cout_glouton, arc_list = RobotGraphe_approche_gloutonne(G)
        list_val_gloutonne.append(cout_glouton)
        list_val_opti.append(cout_optimal)
        list_dist_relativ.append(abs((cout_optimal - cout_glouton)/cout_optimal) )

    num_bins = len(list_dist_relativ)//fact_num_bins
    plt.hist(list_dist_relativ, num_bins, facecolor = 'blue', alpha = 1)
    plt.title("Robot sur Graphe : Distance relative entre méthode gloutonne et optimale")
    plt.ylabel("Nombre d'occurences")
    plt.xlabel("Distance relative")
    plt.savefig(join(save_dir,"Robot_sur_graphe.png"))