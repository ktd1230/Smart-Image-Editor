  
import axios from "axios";

export default axios.create({
  //baseURL: "http://localhost:8000",  
	baseURL: "http://k02c1011.p.ssafy.io:8000",
  headers: {
    "Content-type": "application/json",
  }
});
