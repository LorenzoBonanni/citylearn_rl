import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.sac.policies import Actor, SACPolicy
from custom_rbc import CustomRBC

class RBCGuidedActor(Actor):
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        activation_fn=nn.ReLU,
        use_sde=False,
        log_std_init=-3,
        full_std=True,
        use_expln=False,
        clip_mean=2.0,
        normalize_images=True,
        rbc_controller=None,
        rbc_weight=0.5,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images,
        )
        
        self.rbc_controller = rbc_controller
        self.rbc_weight = rbc_weight
        self.training_steps = 0

    def _predict(self, observation, deterministic=False):
        # Primo passa la predizione alla classe base per ottenere le azioni standard
        actions = super()._predict(observation, deterministic=deterministic)
        
        # Se abbiamo un controllore RBC, integriamolo
        if hasattr(self, 'rbc_controller') and self.rbc_controller is not None and self.training:
            try:
                # Converti le osservazioni in formato numpy per la RBC
                np_obs = observation.cpu().numpy() if isinstance(observation, th.Tensor) else observation
                
                # Ottieni azioni dalla RBC
                with th.no_grad():
                    rbc_actions = self.rbc_controller.predict_multi_obj(np_obs)
                    rbc_actions = th.tensor(rbc_actions, device=actions.device, dtype=actions.dtype)
                    
                    # Peso fisso per test
                    weight = 0.5  # self.rbc_weight
                    
                    # Mix delle azioni: RBC + rete neurale
                    actions = weight * rbc_actions + (1 - weight) * actions
            except Exception as e:
                print(f"Errore nell'integrazione RBC: {e}")
        
        return actions

    def forward(self, obs, deterministic=False):
        """
        Forward pass che combina le azioni della rete neurale con quelle della RBC
        """
        # Usa il features_extractor direttamente invece di extract_features
        features = self.features_extractor(obs)
        latent_pi = self.latent_pi(features)
        
        # Azioni dalla rete neurale
        mean_actions = self.mu(latent_pi)
        
        # Calcola la action_distribution per deterministic=False
        if not deterministic:
            log_std = self.log_std(latent_pi)  # Ottieni il log_std come output
            distribution = self.action_dist.proba_distribution(mean_actions, log_std)
            actions = distribution.get_actions()
            log_prob = distribution.log_prob(actions)
        else:
            actions = mean_actions
            log_prob = None
            
        actions = th.tanh(actions)
        
        # Durante il training, adattare le azioni con un mix di RBC e rete neurale
        if self.training and self.rbc_controller is not None:
            # Converti le osservazioni in formato numpy per la RBC
            np_obs = obs.cpu().numpy() if isinstance(obs, th.Tensor) else obs
            
            # Ottieni azioni dalla RBC
            rbc_actions = []
            try:
                with th.no_grad():
                    # Chiama predict_multi_obj sull'istanza rbc_controller
                    rbc_actions = self.rbc_controller.predict_multi_obj(np_obs)
                    rbc_actions = th.tensor(rbc_actions, device=actions.device, dtype=actions.dtype)
                    
                    # Calcola un peso dinamico che diminuisce gradualmente
                    decay_rate = 0.999  # Regola questo valore per controllare la velocità di decadimento
                    current_weight = self.rbc_weight * (decay_rate ** self.training_steps)
                    
                    # Mix delle azioni: RBC + rete neurale
                    actions = current_weight * rbc_actions + (1 - current_weight) * actions
                    
                    # Incrementa i passi di training
                    self.training_steps += 1
            except Exception as e:
                print(f"Errore nell'utilizzo della RBC: {e}")
        
        # IMPORTANTE: La classe Actor deve restituire solo le azioni 
        # nel contesto _predict, mentre deve restituire la coppia (azioni, log_prob)
        # in altri contesti. Per rendere compatibile il codice con entrambe le situazioni,
        # possiamo utilizzare il flag deterministic come guida.
        if deterministic:
            # Per la predizione deterministica, ritorna solo le azioni
            return actions
        else:
            # Per il training, ritorna la coppia (azioni, log_prob)
            return actions, log_prob
        
    def set_training_mode(self, mode=True):
        """
        Put the policy in training mode.
        """
        self.training = mode
        self.features_extractor.training = mode

class CustomSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        use_sde=False,
        log_std_init=-3,
        use_expln=False,
        clip_mean=2.0,
        features_extractor_class=None,
        features_extractor_kwargs=None,
        normalize_images=True,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None,
        n_critics=2,
        share_features_extractor=False,
        rbc_weight=0.5,
        env=None,
    ):
        # Inizializza i campi personalizzati prima di chiamare il costruttore della classe base
        self.rbc_controller = None
        self.env = env
        self.rbc_weight = rbc_weight
        
        # Inizializza il controllore RBC se possibile
        if env is not None:
            try:
                # Crea una mappa di azioni vuota per la RBC
                action_map = []
                building_map = {}
                
                # Usa ACTIVE_ACTIONS importato dal modulo principale
                from test import ACTIVE_ACTIONS
                
                for action in ACTIVE_ACTIONS:  # Usa la costante definita nel modulo test.py
                    building_map[action] = {hour: 0 for hour in range(1, 25)}
                action_map.append(building_map)
                
                # Crea un'istanza di CustomRBC
                from custom_rbc import CustomRBC
                self.rbc_controller = CustomRBC(env.unwrapped, action_map)
            except Exception as e:
                print(f"Impossibile inizializzare la RBC: {e}")
        
        # Chiama il costruttore della classe base con i parametri corretti
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        # Aggiungi la RBC e il peso come parametri aggiuntivi
        actor_kwargs.update({
            "rbc_controller": self.rbc_controller,
            "rbc_weight": self.rbc_weight
        })
        return RBCGuidedActor(**actor_kwargs).to(self.device)