import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        
        # TODO: Define network parameters and optimizer
        self.learning_rate = .001
        self.dense1 = tf.keras.layers.Dense(state_size*32)
        self.dense2 = tf.keras.layers.Dense(state_size*16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(self.num_actions)

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # TODO: implement this ~
        states = tf.convert_to_tensor(states)
        d1 = self.dense1(states)
        d2 = self.dense2(d1)
        d3 = self.dense3(d2)
        probs = tf.nn.softmax(d3)
        return probs

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        # TODO: implement this
        # Hint: Use gather_nd to get the probability of each action that was actually taken in the episode.
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards)
        episode_length = tf.size(actions)
        
        probs = self.call(states)
        actions = tf.reshape(actions, [episode_length, 1])
        indices = tf.range(episode_length)
        indices = tf.reshape(indices, [episode_length, 1])
        actions = tf.concat((indices, actions), 1)
        probs = tf.gather_nd(probs, actions)
        
        probs = tf.math.log(probs)
        probs = tf.math.multiply(probs, discounted_rewards)
        return -1.0 * tf.math.reduce_sum(probs)