class BaseController:
    def __init__(self):
        pass

    def send_response(self, data=None, message ='', status_code=200):
        if data is None:
            data = []
        else:
            if isinstance(data, dict):
                data = data
            else:
                data = {"data": data}
        return {
            "success": True,
            "message": message,
            "data": data,
            "status_code": status_code,
        }

    def send_error(self, data=None, message ='', status_code=400):
        if data is None:
            data = []
        else:
            if isinstance(data, dict):
                data = data
            else:
                data = {"data": data}
        return {
            "success": False,
            "message": message,
            "data": data,
            "status_code": status_code,
        }