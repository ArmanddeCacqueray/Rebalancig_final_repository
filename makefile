push:
	git add .
	git commit -m "small update"
	git push origin gregoire

pull:
	git pull origin gregoire

activ:
	source venv/bin/activate
	
install:
	pip install -r requirements.txt