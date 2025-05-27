"""
Wrappers di compatibilità per il transfer learning tra ambienti con dimensioni diverse.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ObservationCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper per adattare le osservazioni di dimensione diversa a quella attesa dal modello.
    """
    def __init__(self, env, target_obs_size=16):
        super().__init__(env)
        self.target_obs_size = target_obs_size
        
        # Aggiorna lo spazio delle osservazioni
        low = np.full(target_obs_size, -np.inf, dtype=np.float32)
        high = np.full(target_obs_size, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def _adapt_observation(self, obs):
        """Adatta l'osservazione alla dimensione target."""
        if isinstance(obs, np.ndarray):
            if len(obs.shape) == 1:  # Caso singola osservazione
                if obs.shape[0] > self.target_obs_size:
                    # Tronca l'osservazione
                    return obs[:self.target_obs_size]
                elif obs.shape[0] < self.target_obs_size:
                    # Estende l'osservazione con zeri
                    padded = np.zeros(self.target_obs_size, dtype=obs.dtype)
                    padded[:obs.shape[0]] = obs
                    return padded
            elif len(obs.shape) > 1:  # Caso batch di osservazioni
                if obs.shape[-1] > self.target_obs_size:
                    return obs[..., :self.target_obs_size]
                elif obs.shape[-1] < self.target_obs_size:
                    padded = np.zeros((*obs.shape[:-1], self.target_obs_size), dtype=obs.dtype)
                    padded[..., :obs.shape[-1]] = obs
                    return padded
        return obs
        
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._adapt_observation(observation), info
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._adapt_observation(observation), reward, terminated, truncated, info


class ActionCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper per adattare le azioni di dimensione inferiore a quella attesa dall'ambiente.
    
    Questo wrapper permette di utilizzare un modello addestrato su uno spazio di azioni
    di dimensione minore in un ambiente che richiede azioni di dimensione maggiore.
    """
    def __init__(self, env, source_action_size=9, target_action_size=18, mapping_strategy="duplicate"):
        """
        Inizializza il wrapper.
        
        Args:
            env: L'ambiente da wrappare
            source_action_size: La dimensione delle azioni che il modello produce
            target_action_size: La dimensione delle azioni che l'ambiente si aspetta
            mapping_strategy: Strategia per mappare le azioni 
                              "duplicate" - Duplica le azioni (default)
                              "zero_pad" - Riempie con zeri
                              "mean_pad" - Riempie con il valore medio delle azioni
        """
        super().__init__(env)
        self.source_action_size = source_action_size
        self.target_action_size = target_action_size
        self.mapping_strategy = mapping_strategy
        
        # Salva lo spazio delle azioni originale
        self.original_action_space = env.action_space
        
        # Aggiorna lo spazio delle azioni per accettare lo spazio delle azioni più piccolo
        if isinstance(env.action_space, spaces.Box):
            low = env.action_space.low[:source_action_size]
            high = env.action_space.high[:source_action_size]
            self.action_space = spaces.Box(low=low, high=high, dtype=env.action_space.dtype)
        else:
            # Se lo spazio delle azioni non è Box, solleviamo un'eccezione
            raise ValueError("ActionCompatibilityWrapper supporta solo spazi di azioni Box")
    
    def step(self, action):
        """
        Adatta l'azione alla dimensione richiesta dall'ambiente e esegue un passo.
        """
        # Converti l'azione a numpy array per sicurezza
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        # Expandi l'azione alla dimensione desiderata
        expanded_action = self._expand_action(action)
        return self.env.step(expanded_action)
    
    def _expand_action(self, action):
        """
        Espandi un'azione di dimensione source_action_size a dimensione target_action_size.
        
        Args:
            action: L'azione da espandere
            
        Returns:
            Un'azione espansa di dimensione target_action_size
        """
        if self.mapping_strategy == "duplicate":
            # Se abbiamo 9 azioni e ne servono 18, duplichiamo le 9 azioni
            if self.target_action_size % self.source_action_size == 0:
                # Caso perfettamente divisibile
                repeat_factor = self.target_action_size // self.source_action_size
                return np.repeat(action, repeat_factor)
            else:
                # Caso non perfettamente divisibile, ripetiamo e tronchiamo
                expanded = np.tile(action, (self.target_action_size // self.source_action_size) + 1)
                return expanded[:self.target_action_size]
        
        elif self.mapping_strategy == "zero_pad":
            # Aggiungi zeri per completare la dimensione richiesta
            expanded = np.zeros(self.target_action_size, dtype=action.dtype)
            expanded[:min(len(action), self.target_action_size)] = action[:min(len(action), self.target_action_size)]
            return expanded
        
        elif self.mapping_strategy == "mean_pad":
            # Riempi con il valore medio delle azioni
            expanded = np.full(self.target_action_size, np.mean(action), dtype=action.dtype)
            expanded[:min(len(action), self.target_action_size)] = action[:min(len(action), self.target_action_size)]
            return expanded
        
        else:
            raise ValueError(f"Strategia di mappatura non valida: {self.mapping_strategy}")