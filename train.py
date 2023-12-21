import torch

def train(net, train_loader, test_loader, criterion, optimizer, num_epochs, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        net.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data, target = data.float(), target.float()
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        net.eval()  # 确保模型处于评估模式
        test_loss = 0
        with torch.no_grad():
            for data_t, target_t in test_loader:
                data_t, target_t = data_t.to(device), target_t.to(device)
                data_t, target_t = data_t.float(), target_t.float()
                output_t = net(data_t)
                test_loss += criterion(output_t, target_t).item()

            test_loss /= len(test_loader)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}')
    torch.save(net.state_dict(), 'models/' + model_name)
    print("Model saved.")