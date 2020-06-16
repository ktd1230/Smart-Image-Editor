  
import axios from "axios";

export default axios.create({
  baseURL: "http://localhost:8000",  
  //baseURL: "http://i02c104.p.ssafy.io:8000",
  headers: {
    "Content-type": "application/json",
  }
});
