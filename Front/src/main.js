import Vue from 'vue'
import App from './App.vue'
import './plugins/element.js'
import vuetify from './plugins/vuetify';
import 'material-design-icons-iconfont/dist/material-design-icons.css'
import router from './router'

import AOS from "aos"
import "aos/dist/aos.css"
import store from './store'

Vue.config.productionTip = false
Vue.use(vuetify);

new Vue({
  created(){
    AOS.init();
  },
  
  vuetify,
  router,
  store,
  render: h => h(App)
}).$mount('#app')
