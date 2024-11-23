import importlib

module_name = f"opt_data_loader.model_train_opt_data_loader.py"  # script1.py должен быть точкой входа
module = importlib.import_module(module_name)
module.main()  # вызываем функцию main() или любую другую точку входа
