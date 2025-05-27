from stable_baselines3.common.callbacks import BaseCallback
from custom_rbc import CustomRBC
from constants import ACTIVE_ACTIONS  # Importiamo da constants invece che da config
import numpy as np
import torch as th

class RBCPureCallback(BaseCallback):
    """
    Un callback che inizialmente usa solo azioni RBC per un certo numero di step,
    poi passa completamente alle azioni del modello appreso (SAC).
    Invece di miscelare le azioni, questo approccio usa prima 100% RBC e poi 100% modello.
    """
    def __init__(self, env, rbc_steps=1000, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.rbc_steps = rbc_steps    # Numero di step in cui usare solo RBC
        self.steps = 0                # Contatore per monitorare i passi
        self.use_rbc = True           # Flag per indicare se usare RBC
        
        # Inizializza RBC
        try:
            action_map = []
            building_map = {}
            for action in ACTIVE_ACTIONS:
                building_map[action] = {hour: 0 for hour in range(1, 25)}
            action_map.append(building_map)
            self.rbc = CustomRBC(env.unwrapped, action_map)
            self.rbc_initialized = True
            if verbose > 0:
                print("RBC inizializzata con successo")
        except Exception as e:
            print(f"Errore nell'inizializzazione della RBC: {e}")
            self.rbc_initialized = False
    
    def _on_step(self):
        self.steps += 1
        
        # Controlla se è il momento di passare al modello appreso
        if self.steps >= self.rbc_steps and self.use_rbc:
            self.use_rbc = False
            if self.verbose > 0:
                print(f"Passo {self.steps}: passaggio da RBC a policy appresa")
        
        # Stampa di debug ogni 1000 passi
        if self.verbose > 0 and self.steps % 1000 == 0:
            mode = "RBC" if self.use_rbc else "policy appresa"
            print(f"Passo {self.steps}: uso {mode}")
        
        return True
    
    def _on_rollout_start(self):
        # Salva il riferimento al modello per poter accedere alla policy
        self.model = self.locals["self"]
    
    def _on_rollout_end(self):
        # Non fare nulla se RBC non è inizializzata o se non dobbiamo usare RBC
        if not self.rbc_initialized or not self.use_rbc:
            return
        
        # Durante la fase di raccolta esperienze, sostituisci le azioni con quelle RBC
        if hasattr(self.model, "replay_buffer") and hasattr(self.model.replay_buffer, "buffer_size"):
            # Verifica se ci sono dati nel buffer usando proprietà sicure
            try:
                # Usa il metodo sample con gestione degli errori
                buffer_size = self.model.replay_buffer.buffer_size
                if buffer_size > 0 and self.model.replay_buffer.pos > 0:
                    # Determina quanti campioni 
                    sample_size = min(512, self.model.replay_buffer.pos)
                    
                    # Recupera le ultime esperienze raccolte
                    replay_data = self.model.replay_buffer.sample(sample_size)
                    observations = replay_data.observations
                    actions = replay_data.actions
                    
                    # Stampa debug per controllare le dimensioni
                    if self.verbose > 1:
                        print(f"Forma observations: {observations.shape}")
                        print(f"Forma actions: {actions.shape}")
                    
                    # Ottieni le azioni dalla RBC per le stesse osservazioni
                    np_obs = observations.cpu().numpy() if isinstance(observations, th.Tensor) else observations
                    
                    # Assicurati che le dimensioni siano corrette
                    if len(np_obs.shape) == 2:  # batch, features
                        with th.no_grad():
                            try:
                                # Predici azioni RBC
                                rbc_actions = self.rbc.predict_multi_obj(np_obs)
                                
                                # Stampa informazioni sulle forme degli array per debug
                                if self.verbose > 0:
                                    print(f"Forma rbc_actions prima del processing: {np.array(rbc_actions).shape}")
                                
                                # Adatta le dimensioni delle azioni RBC a quelle attese
                                if len(rbc_actions) == 1 and isinstance(rbc_actions[0], list):
                                    # Caso di ambiente con un singolo edificio ma batch di osservazioni
                                    if sample_size > 1:
                                        # Replica l'azione per tutte le osservazioni nel batch
                                        rbc_actions = np.array([rbc_actions[0] for _ in range(sample_size)])
                                else:
                                    # Assicurati che rbc_actions sia nella forma corretta (batch, action_dim)
                                    rbc_actions = np.array(rbc_actions)
                                    if len(rbc_actions.shape) == 3 and rbc_actions.shape[1] == 1:
                                        # Caso (batch, 1, action_dim) -> (batch, action_dim)
                                        rbc_actions = rbc_actions.squeeze(1)
                                
                                if self.verbose > 0:
                                    print(f"Forma rbc_actions dopo processing: {np.array(rbc_actions).shape}")
                                    print(f"Forma actions target: {actions.shape}")
                                
                                # Converti le azioni RBC in tensori
                                rbc_tensor = th.tensor(
                                    rbc_actions, 
                                    device=actions.device, 
                                    dtype=actions.dtype
                                )
                                
                                # Adatta esplicitamente la forma
                                if len(actions.shape) == 3 and len(rbc_tensor.shape) == 2:
                                    # Se actions è (batch, 1, action_dim) e rbc_tensor è (batch, action_dim)
                                    # Aggiungi una dimensione in mezzo
                                    rbc_tensor = rbc_tensor.unsqueeze(1)
                                    if self.verbose > 0:
                                        print(f"Forma rbc_tensor dopo unsqueeze: {rbc_tensor.shape}")
                                
                                # Assicurati che le forme siano compatibili 
                                if rbc_tensor.shape == actions.shape:
                                    # Sostituisci completamente le azioni con quelle RBC
                                    if hasattr(self.model.replay_buffer, "actions"):
                                        last_idx = min(sample_size, len(self.model.replay_buffer.actions))
                                        self.model.replay_buffer.actions[-last_idx:] = rbc_tensor
                                    
                                    if self.verbose > 0:
                                        print(f"Azioni del buffer sostituite completamente con RBC")
                                else:
                                    # Tenta di adattare le dimensioni per la sostituzione
                                    try:
                                        if len(actions.shape) == 3 and len(rbc_tensor.shape) == 2:
                                            # Reshape rbc_tensor per evitare l'errore di broadcasting
                                            rbc_tensor = rbc_tensor.reshape(actions.shape)
                                            if hasattr(self.model.replay_buffer, "actions"):
                                                last_idx = min(sample_size, len(self.model.replay_buffer.actions))
                                                self.model.replay_buffer.actions[-last_idx:] = rbc_tensor
                                            if self.verbose > 0:
                                                print(f"Azioni sostituite dopo reshape a {rbc_tensor.shape}")
                                        else:
                                            if self.verbose > 0:
                                                print(f"Errore: dimensioni non compatibili - rbc_tensor: {rbc_tensor.shape}, actions: {actions.shape}")
                                                print("Non è stato possibile sostituire le azioni.")
                                    except Exception as reshape_err:
                                        if self.verbose > 0:
                                            print(f"Errore nel tentativo di reshape: {reshape_err}")
                            except Exception as e:
                                if self.verbose > 0:
                                    print(f"Errore nella generazione o sostituzione delle azioni: {e}")
                    else:
                        if self.verbose > 0:
                            print(f"Forma delle osservazioni non supportata: {np_obs.shape}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Errore nell'accesso al replay buffer: {e}")
                    print(f"Tipo del replay buffer: {type(self.model.replay_buffer)}")