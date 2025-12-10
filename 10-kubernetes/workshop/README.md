### K8s Workshop

#### To create cluster
```bash
kind create cluster --name <cluster-name>
```

#### To list cluster
```bash
kind get clusters
```

#### To load image to Kind
```bash
kind load docker-image <image-name>:<tag> --name <cluster-name>
```

#### To clean up
```bash
kind delete cluster --name <your-cluster-name>
```