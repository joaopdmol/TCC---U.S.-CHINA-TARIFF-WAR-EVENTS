# GitHub Push Instructions

O projeto ja esta preparado localmente para versionamento.

## 1. Criar um repositorio no GitHub

Crie um repositorio vazio no GitHub, sem README inicial, por exemplo:

- `tcc-trade-war`

Copie a URL do repositorio, algo como:

```bash
https://github.com/SEU_USUARIO/tcc-trade-war.git
```

## 2. Subir este projeto ao GitHub

Se voce estiver em uma maquina com `git` instalado, entre na pasta do projeto e rode:

```bash
git remote add origin https://github.com/SEU_USUARIO/tcc-trade-war.git
git branch -M main
git push -u origin main
```

Se o remote `origin` ja existir, use:

```bash
git remote set-url origin https://github.com/SEU_USUARIO/tcc-trade-war.git
git branch -M main
git push -u origin main
```

## 3. Baixar depois no Mac

No Mac, rode:

```bash
git clone https://github.com/SEU_USUARIO/tcc-trade-war.git
cd tcc-trade-war
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Continuar o trabalho no Mac

Depois de clonar no Mac:

```bash
git pull origin main
```

E, para subir novas alteracoes depois:

```bash
git add .
git commit -m "mensagem clara"
git push origin main
```
