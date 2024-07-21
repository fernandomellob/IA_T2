# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
        Retorna o valor Q(state,action).
        Retorna 0.0 se nunca vimos este par estado-ação antes.
        """
        # Usa o método get() do dicionário com valor padrão 0.0
        return self.qValues.get((state, action), 0.0)

    def computeValueFromQValues(self, state):
        """
        Retorna o valor máximo de Q(state,action) para todas as ações legais.
        Se não houver ações legais, retorna 0.0 (estado terminal).
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0

        # Usa getQValue para acessar os valores Q
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
        Computa a melhor ação a ser tomada em um estado.
        Retorna None se não houver ações legais (estado terminal).
        Quebra empates aleatoriamente.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        # Usa um dicionário para agrupar ações por valor Q
        actionValues = {}
        for action in legalActions:
            q_value = self.getQValue(state, action)
            if q_value not in actionValues:
                actionValues[q_value] = []
            actionValues[q_value].append(action)

        # Encontra o valor Q máximo
        bestQValue = max(actionValues.keys())

        # Retorna uma ação aleatória entre as melhores ações
        return random.choice(actionValues[bestQValue])

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        features = self.featExtractor.getFeatures(state, action)
        q_value = 0.0
        for feature, value in features.items():
            q_value += value * self.weights[feature]

        return q_value

    def update(self, state, action, nextState, reward):
        """
        Atualiza os pesos do agente com base na transição observada.
        
        Parâmetros:
        state -- O estado atual
        action -- A ação tomada
        nextState -- O próximo estado após a ação
        reward -- A recompensa recebida pela transição
        """
        
        # Extrai as features do par estado-ação atual
        features = self.featExtractor.getFeatures(state, action)
        
        # Calcula o Q-valor atual para o par estado-ação
        current_q_value = self.getQValue(state, action)
        
        # Calcula o valor máximo esperado do próximo estado
        next_max_q_value = self.getValue(nextState)
        
        # Calcula o erro de diferença temporal (TD error)
        # TD error = (recompensa + desconto * melhor próximo valor) - valor atual
        td_error = reward + (self.discount * next_max_q_value) - current_q_value
        
        # Atualiza os pesos para cada feature
        for feature, value in features.items():
            # Fórmula de atualização: peso += taxa_aprendizado * td_error * valor_feature
            self.weights[feature] += self.alpha * td_error * value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
