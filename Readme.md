# demo-setup-cicd-ml

## 1. Initial project

```
https://github.com/hellsing032/demo-cicd-cc31.git
```
## 2. Git workflow

- 1. Pull any update on main branch
```
git pull
```

- 2. Setup .venv on windows (Setup .venv localhost)
```
python -m venv .venv

.venv\Script\active
```

- 2.1 Setup .venv on mac & linux
```
python -m venv .venv

source .venv/Scripts/activate
``` 

- 3. (kerjain fiturnya)
- P.S (Kalau mau untuk memasukan depedency library pada requirements.tx)
```
pip freeze > requirements.txt
```

- 4. add dan commit

```
git add .
git commit -m "Nama Feature - Nama Perubahan"

git push origin main
```

test test