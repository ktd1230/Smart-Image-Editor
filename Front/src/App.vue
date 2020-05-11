<template>
  <v-app>
    <toolbar></toolbar>
    <loading class="fixed" :loading="loadingStatus"></loading>
    <router-view :key="$route.fullPath">
    </router-view>
  </v-app>
</template>

<script>
import bus from './utiles/bus.js';
import Toolbar from './components/home/toolbar'
import Loading from './components/home/loading.vue';

    export default {
        components: {
          Toolbar,
          Loading,
        },
        data: ()=> ({
          loadingStatus: false,
        }),
        methods: {
          startLoading() {
            this.loadingStatus = true;
          },
          endLoading() {
            this.loadingStatus = false;
          }
        },
        created() {
          bus.$on('start:loading', this.startLoading);
          bus.$on('end:loading', this.endLoading);
        },
        beforeDestroy() {
          bus.$off('start:loading');
          bus.$off('end:loading');
        },
    }
</script>
<style scoped>
.fixed {
    position: fixed;
    right: 50%;
    bottom: 60%;
    z-index: 1;
}
</style>
