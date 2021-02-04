class Utils:
    def checkGradDep(*tensors) -> bool:
        """
        Checks if any of the given tensors have a non-required-gradient switch
        """
        for i in tensors:
            if not i.requires_grad:
                return False
        return True