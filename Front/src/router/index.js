import Vue from 'vue';
import VueRouter from 'vue-router';
import Home from '../components/home';
import Mybook from '../components/home/mybook';
import Result from '../components/book/result';
import Rank from '../components/home/rank';
import Detail from '../components/home/detail';
import Test from '../components/test/test'

Vue.use(VueRouter)

export default new VueRouter({
  mode: "history",
  routes : [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/mybook',
    name: 'mybook',
    component: Mybook
  },
  {
    path: '/result',
    name: 'result',
    component: Result,
    props: true,
  },
  {
    path: '/rank',
    name: 'rank',
    component: Rank
  },
  {
    path: '/detail',
    name: 'detail',
    component: Detail,
    props: true,
  },
  {
    path:'/test',
    name: 'detail',
    component:Test
  }
]
})
