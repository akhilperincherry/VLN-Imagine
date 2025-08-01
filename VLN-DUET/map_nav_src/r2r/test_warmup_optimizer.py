

class TestWarmupRoutine():
    def __init__(self):
        pass

    def validate_all_params_are_accounted(self, model, optimizer): #in optimizer
        all_vln_bert_params = list(model.parameters())
        optimizer_params = []
        for group in optimizer.param_groups:
            optimizer_params.extend(group['params'])
        # Ensure that all model parameters are covered by optimizer groups
        assert set(all_vln_bert_params) == set(optimizer_params), \
            "Mismatch between vln_bert parameters and optimizer parameters!"

    def ensure_no_duplicate_params(self, optimizer):
        # Get parameters from each optimizer group
        group_params = [set(p for p in group['params']) for group in optimizer.param_groups]

        # Ensure that no parameters overlap across groups
        for i in range(len(group_params)):
            for j in range(i + 1, len(group_params)):
                assert group_params[i].isdisjoint(group_params[j]), f"Parameter overlap between optimizer groups {i} and {j}!"

    def print_named_parameters(self, model): #model = listner.vln_bert
        names = []
        for name, param in model.named_parameters():
            names.append(name)
        return names

    def get_param_lr(self, model, optimizer, name_substring):
        """Returns learning rate(s) for parameters containing `name_substring` in their name."""
        lrs = []
        # Get the names of the parameters and their learning rates
        named_params = dict(model.named_parameters())
        # pdb.set_trace()
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                # Find the name corresponding to the parameter tensor
                param_name = [name for name, p in named_params.items() if p is param]
                if param_name and name_substring in param_name[0]:
                    lrs.append(param_group['lr'])
        return lrs

    # Get learning rate of the rest of the model
    def get_rest_of_model_lr(self, model, optimizer, excluded_substrings):
        """Returns learning rate(s) for parameters not in excluded_substrings."""
        lrs = []
        named_params = dict(model.named_parameters())
        
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                param_name = [name for name, p in named_params.items() if p is param]
                if param_name and not any(substring in param_name[0] for substring in excluded_substrings):
                    lrs.append(param_group['lr'])
        return lrs

    def get_params_trainable_status(self, model, name_substring=None):
        """
        Get the trainable status of all parameters in the model that contain the given name_substring.
        If name_substring is None, it returns the status for all parameters.
        """
        trainable_params = {}
        
        for name, param in model.named_parameters():
            if name_substring is None or name_substring in name:
                trainable_params[name] = param.requires_grad
                
        return trainable_params

    def get_remaining_params_trainable_status(self, model, excluded_substrings):
        """Get trainable status for the parameters not in excluded_substrings."""
        trainable_params = {}
        
        for name, param in model.named_parameters():
            if not any(substring in name for substring in excluded_substrings):
                trainable_params[name] = param.requires_grad
                
        return trainable_params

    def are_all_params_trainable(self, params):
        """Returns True if all params are trainable (requires_grad=True), False otherwise."""
        return all(is_trainable for is_trainable in params.values())

    def parameter_count_matches_model_optimizer_groups(self, model, optimizer):
        # Total parameters in the model
        total_params = sum(p.numel() for p in model.parameters())

        # Parameters in each optimizer group
        params_in_optimizer = sum(p.numel() for group in optimizer.param_groups for p in group['params'])

        # print(f"Total parameters in the model: {total_params}")
        # print(f"Total parameters in optimizer groups: {params_in_optimizer}")
        assert total_params == params_in_optimizer, "Mismatch between model parameters and optimizer parameters!"

    def is_named_param_trainable(self, model, param_name):
        """Checks if a particular parameter is trainable (requires_grad)."""
        # usage: is_param_trainable(listner.vln_bert, 'vln_bert.encoder.layer.1.intermediate.dense.bias')
        # is_param_trainable(listner.vln_bert, 'vln_bert.imagine_embeddings.type_embedding.weight')
        
        # Get the parameter from the model by name
        for name, param in model.named_parameters():
            if name == param_name:
                return param.requires_grad
        return None  # Return None if the parameter is not found
    
