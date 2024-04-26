from server.models.modelCNN import ModelCNN


model = ModelCNN()
for name, param in model.named_parameters():
    print(name)