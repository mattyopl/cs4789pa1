import numpy as np


class InfiniteHorizon:
    def __init__(self, MDP):
        self.R = MDP.R  # |A|x|S|
        self.P = MDP.P  # |A|x|S|x|S|
        self.discount = MDP.discount
        self.nStates = MDP.nStates
        self.nActions = MDP.nActions

    ####Helpers####
    def extractRpi(self, pi):
        '''
        Returns R(s, pi(s)) for all states. Thus, the output will be an array of |S| entries. 
        This should be used in policy evaluation and policy iteration. 

        Parameter pi: a deterministic policy
        Precondition: An array of |S| integers, each of which specifies an action (row) for a given state s.

        HINT: Given an m x n matrix A, the expression

        A[row_indices, col_indices] (len(row_indices) == len(col_indices))

        returns a matrix of size len(row_indices) that contains the elements

        A[row_indices[i], col_indices[i]] in a row for all indices i.
        '''
        return self.R[pi, np.arange(len(self.R[0]))]

    def extractPpi(self, pi):
        '''
        Returns P^pi: This is a |S|x|S| matrix where the (i,j) entry corresponds to 
        P(j|i, pi(i))


        Parameter pi: a deterministic policy
        Precondition: An array of |S| integers
        '''
        return self.P[pi, np.arange(len(self.P[0]))]

    ####Value Iteration###
    def computeVfromQ(self, Q, pi):
        '''
        Returns the V function for a given Q function corresponding to a deterministic policy pi. Remember that

        V^pi(s) = Q^pi(s, pi(s))

        Parameter Q: Q function
        Precondition: An array of |S|x|A| numbers

        Parameter pi: Policy
        Preconditoin: An array of |S| integers
        '''
        # TODO 1
        # For each state, we want to calculate the V and package that into an array
        # V = np.zeros((self.nStates))
        # for s in range(self.nStates):  
        #     V[s] = Q[s, pi[s]]
        V = Q[np.arange(len(pi)), pi] # iterating over the individual states (using np.arrange [which just lists out 0, ..., len(pi)-1], and pi [to get the action choice of the policy for the state])
        return V



    def computeQfromV(self, V):
        '''
        Returns the Q function given a V function corresponding to a policy pi. The output is an |S|x|A| array.

        Use the bellman equation for Q-function to compute Q from V.

        Parameter V: value function
        Precondition: An array of |S| numbers
        '''
        # TODO 2

        Q = (self.R + self.discount * self.P @ V).T # batches out the calculation into |A| batches of |S|x|S| times |S| -> |A|x|S| results.
        # Q = np.zeros((self.nStates, self.nActions)) 

        # for s in range(self.nStates):  # Iterate over states
        #     for a in range(self.nActions):  # Iterate over actions
        #         expected_value = np.sum([self.P[a, s, s_prime] * V[s_prime] for s_prime in range(self.nStates)])  

        #         Q[s, a] = self.R[a, s] + self.discount * expected_value

        return Q


    def extractMaxPifromQ(self, Q):
        '''
        Returns the policy pi corresponding to the Q-function determined by 

        pi(s) = argmax_a Q(s,a)


        Parameter Q: Q function 
        Precondition: An array of |S|x|A| numbers
        '''
        # TODO 3

        # Iterating over all the possible states, we want to know the maximal action to take. Just take argmax over axis=1
        return  np.argmax(Q, axis=1)


    def extractMaxPifromV(self, V):
        '''
        Returns the policy corresponding to the V-function. Compute the Q-function
        from the given V-function and then extract the policy following

        pi(s) = argmax_a Q(s,a)

        Parameter V: V function 
        Precondition: An array of |S| numbers
        '''
        # TODO 4
        Q = self.computeQfromV(V)
        pi = self.extractMaxPifromQ(Q)
        return pi


    def valueIterationStep(self, Q):
        '''
        Returns the Q function after one step of value iteration. The input
        Q can be thought of as the Q-value at iteration t. Return Q^{t+1}.

        Parameter Q: value function 
        Precondition: An array of |S|x|A| numbers
        '''
        # TODO 5
        Q_prime = np.zeros((self.nStates, self.nActions))

        # # # Compute the maximum Q-values for each state
        # max_Q = np.max(Q, axis=1)

        # max_Q = max_Q[np.newaxis, np.newaxis, :]  # Shape: (1, 1, nStates). Resizing so shapes agree in the sum...


        # e_v = np.sum(self.P * max_Q, axis=2) # Expected value. Summing over the 3rd axis (other two are absorbed)
        # Q_prime = (self.R + self.discount * e_v).T # Taking transpose because Rep is |A|x|S| and we need |S|x|A|

        for s in range(self.nStates):  # Iterate over states
            for a in range(self.nActions):  # Iterate over actions
                # Compute the expected value term
                expected_value = np.sum(self.P[a, s, :] * np.max(Q, axis=1))  # Sum over s'
                # Compute Q^{t+1}(s, a)
                Q_prime[s, a] = self.R[a, s] + self.discount * expected_value

        return Q_prime


    def valueIteration(self, initialQ, tolerance=0.01):
        '''
        This function runs value iteration on the input initial Q-function until 
        a certain tolerance is met. Specifically, value iteration should continue to run until 
        ||Q^t-Q^{t+1}||_inf <= tolerance. Recall that for a vector v, ||v||_inf is the maximum 
        absolute element of v. 


        This function should return the policy, value function, number
        of iterations required for convergence, and the end epsilon where the epsilon is 
        ||Q^t-Q^{t+1}||_inf. 

        Parameter initialQ:  Initial value function
        Precondition: array of |S|x|A| entries

        Parameter tolerance: threshold threshold on ||Q^t-Q^{t+1}||_inf
        Precondition: Float >= 0 (default: 0.01)
        '''
        # TODO 6
        #Placeholder, replace with your code. 
        iterId = 0
        epsilon = np.inf
        Q_t = initialQ
        while True:
            Q_t_1 = self.valueIterationStep(Q_t)
            epsilon = np.max(np.abs(Q_t - Q_t_1))  # Directly compute infinity norm
            Q_t = Q_t_1
            iterId+=1
            if epsilon <= tolerance:
                break
            
        # Now we should have our Q_t that passes the threshold
        # We just need to return the policy, value func, num of iterations, and end epsilon
        pi = self.extractMaxPifromQ(Q_t)
        V = self.computeVfromQ(Q_t, pi)
        return pi, V, iterId, epsilon

    ### EXACT POLICY EVALUATION  ###
    def exactPolicyEvaluation(self, pi):
        '''

        Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma P^pi V^pi

        Return the value function

        Parameter pi: Deterministic policy 
        Precondition: array of |S| integers

        '''
        # TODO 7
        p_pi = self.extractPpi(pi)
        r_pi = self.extractRpi(pi)
        I = np.eye(self.nStates, self.nStates)
        # V = (I-gamma*Pi)^-1 R -> (I-gamma*Pi) * V = R -> Ax = b
        A = I-self.discount*p_pi
        v_pi = np.linalg.solve(A, r_pi)
        return v_pi.T



    ### APPROXIMATE POLICY EVALUATION ###
    def approxPolicyEvaluation(self, pi, tolerance=0.01):
        '''
        Evaluate a policy using approximate policy evaluation. Like value iteration, approximate 
        policy evaluation should continue until ||V_n - V_{n+1}||_inf <= tolerance. 

        Return the value function, number of iterations required to get to exactness criterion, and final epsilon value.

        Parameter pi: Deterministic policy 
        Precondition: array of |S| integers

        Parameter tolerance: threshold threshold on ||V^n-V^n+1||_inf
        Precondition: Float >= 0 (default: 0.01)
        '''
        # TODO 8
        V = np.zeros(self.nStates) # init v_pi_0 with 0s
        epsilon = np.inf
        n_iters = 0

        p_pi = self.extractPpi(pi)
        r_pi = self.extractRpi(pi)

        while(True):
            # V_t_1 = r_pi + self.discount * p_pi @ V
            V_t_1 = r_pi + self.discount * np.dot(p_pi, V)
            epsilon = np.max(np.abs(V_t_1 - V))
            n_iters += 1
            V = V_t_1
            if epsilon <= tolerance:
                break
            

        return V, n_iters, epsilon

    def policyIterationStep(self, pi, exact):
        '''
        This function runs one step of policy evaluation, followed by one step of policy improvement. Return
        pi^{t+1} as a new numpy array. Do not modify pi^t.

        Parameter pi: Current policy pi^t
        Precondition: array of |S| integers

        Parameter exact: Indicate whether to use exact policy evaluation 
        Precondition: boolean
        '''
        # TODO 9
        #  Performing one step of policy eval
        V = self.exactPolicyEvaluation(pi) if exact else self.approxPolicyEvaluation(pi)[0]

        pi = self.extractMaxPifromV(V)
        # Now one step of policy improvement
        return pi

    def policyIteration(self, initial_pi, exact):
        '''

        Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a Q^pi(s,a)).


        This function should run policyIteration until convergence where convergence 
        is defined as pi^{t+1}(s) == pi^t(s) for all states s.

        Return the final policy, value-function for that policy, and number of iterations
        required until convergence.

        Parameter initial_pi:  Initial policy
        Precondition: array of |S| entries

        Parameter exact: Indicate whether to use exact policy evaluation 
        Precondition: boolean

        '''
        # TODO 10
        #Placeholder, replace with your code. 
        iterId = 0
        pi = initial_pi
        V = np.zeros(self.nStates)

        while True:
            pi_t_1 = self.policyIterationStep(pi, exact)
            same = np.array_equal(pi, pi_t_1)
            pi = pi_t_1
            iterId+=1
            if(same):
                break

        # Now calculating the final value function for the policy...
        if(exact):
            V = self.exactPolicyEvaluation(pi)
        else:
            V, _, _ = self.approxPolicyEvaluation(pi)
        
        return pi, V, iterId
