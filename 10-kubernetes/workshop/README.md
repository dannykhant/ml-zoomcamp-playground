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
kubectl delete -f k8s-deploy.yaml

kind delete cluster --name <cluster-name>
```

#### To find node IP
```bash
kubectl get nodes -o wide
```

#### To find node port of the service
```bash
kubectl get svc -n <namespace>
```

#### To install metrics-server for HPA
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

#### To patch matrics-server for Kind
```bash
kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```
