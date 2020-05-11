<template>
  <v-app  id="content">
    <v-layout column="column" wrap="wrap" class="my-12" align-center="align-center" style="flex:0;">
        <v-flex xs12="xs12" sm4="sm4" class="mt-4">
            <div class="text-center">
                <h2 id="font2">도서 랭킹 목록</h2>
                <br>
                <span class="subheading font1">다른 사용자의 도서를 확인할 수 있습니다.</span><br/>
                <span class="subheading font1">다양한 스토리를 공유해보세요.</span>
            </div>
        </v-flex>
    </v-layout>
    <v-layout column="column">
        <div class="d-flex col-sm-8 col-md-6 col-lg-4 mx-auto">
            <v-text-field 
                placeholder="검색"
                v-model="keyword"
                filled
                rounded
                dense
                @keydown.enter="search(keyword)"
                :append-outer-icon="'mdi-send'">
            </v-text-field>
        </div>
    </v-layout>
      <div class="bg row d-flex justify-center">
        <br>
        <div v-for="(story, i) in storys" :key="i" class="col-md-5 col-lg-3" style="margin:20px;">
            <img src="../../assets/img/medal.png" class="col-3" style="position:absolute;z-index:1;">
            <h1 @click="like(story.id)">
                <v-btn icon color="#C62828" style="outline:0;" class="d-flex align-center mx-auto">
                    <v-icon v-if="like_stories.includes(story.id)" style="color:red">mdi-heart</v-icon>
                    <v-icon v-else style="color:red">mdi-heart-outline</v-icon>
                    <p class="my-auto" style="color:red"> ({{stories_like_cnt[story.id]}})</p>
                </v-btn>
            </h1>
            <v-hover>
                <template v-slot:default="{ hover }" style="position:absolute;z-index:0;">
                <div class="card text-white" style="background-color:unset;">
                    <img class="card-img" height="230" :src="`http://13.124.246.175:8000/media/${story.firstimage}`" alt="" style="width: auto;object-fit:cover;">
                    <div class="card-img-overlay text-center px-0">
                        <p class="text-light black">{{story.user.username}}</p>
                        <h3 v-if="story.title" class="mt-4 card-title black">{{story.title}}</h3>
                        <h3 v-else class="mt-4 card-title">...</h3>
                        <p class="py-1 subheading black card-text" style="margin-top:100px;">{{story.create_at}}</p>
                        <v-fade-transition>
                            <v-overlay
                                v-if="hover"
                                absolute
                                color="#036358"
                            >
                                <v-btn @click="detail(story)">자세히 보기</v-btn>
                            </v-overlay>
                        </v-fade-transition>
                    </div>
                </div>
                </template>
            </v-hover>
        </div>
      </div>
  </v-app>
</template>

<script>
import axios_common from '../../axios_common';
import { mapGetters } from 'vuex';

export default {
    computed: {
        ...mapGetters([
            'isAuthenticated',
            'requestHeader',
            'userId',
            'username'
        ])
    },
    data() {
        return {
            storys : [],
            first_image : [],
            like_stories: [],
            stories_like_cnt: {},
            keyword:'',
            books: [
                {
                    title: "책1",
                    img: 'https://i.pinimg.com/564x/7b/8f/1e/7b8f1e3287a38700d88d1596c1144c9b.jpg',
                    date: '2020-04-20'
                },
                {
                    title: '책2',
                    img: 'https://elib.seoul.go.kr/resources/images/YES24/Msize/8638926M.jpg',
                    date: '2020-04-22'
                },
                {
                    title: '책3',
                    img: 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTOfEu4PQtrmmW5gkjA0RBT-8rN3eSBKVZnLdZpVxTveJozU7pm&usqp=CAU',
                    date: '2020-04-22'
                },
                {
                    title: "책1",
                    img: 'https://i.pinimg.com/564x/7b/8f/1e/7b8f1e3287a38700d88d1596c1144c9b.jpg',
                    date: '2020-04-20'
                },
                {
                    title: '책2',
                    img: 'https://elib.seoul.go.kr/resources/images/YES24/Msize/8638926M.jpg',
                    date: '2020-04-22'
                },
                {
                    title: '책3',
                    img: 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTOfEu4PQtrmmW5gkjA0RBT-8rN3eSBKVZnLdZpVxTveJozU7pm&usqp=CAU',
                    date: '2020-04-22'
                }
            ],
        }
    },   
    methods: {
        like(id){
            if (!this.isAuthenticated) {
                alert("로그인 해주세요.");
            }else {
                axios_common.get(`/sub3/like/${id}`, this.requestHeader)
                .then( response => {
                    console.log(response.data)
                    const onLike = response.data.on_like

                    if(onLike){
                        this.like_stories.push(id);
                        this.stories_like_cnt[id] += 1;
                        // console.log(this.like_stories)
                    }else{
                        const idx = this.like_stories.indexOf(id);
                        if (idx > -1) this.like_stories.splice(idx, 1);
                        this.stories_like_cnt[id] -= 1;
                    }
                })
                .catch(error => console.log(error))
            }
        },
        search(keyword){
            axios_common.post('/sub3/search/', {'keyword':keyword})
            .then(response => {
                console.log(response.data)
                this.storys = response.data
            })
            .catch(error => console.log(error))
        },
        detail(story) {
            this.$router.push({ name: 'detail', params: {story: story}});
        }
    },
    mounted() {
        // console.log(this.isAuthenticated)
        axios_common.get('/sub3/allstory/')
            .then(response => {
                // console.log(response.data)
                this.storys = response.data
                for (let i=0;i<response.data.length;i++){
                    this.stories_like_cnt[response.data[i].id] = response.data[i].like_users.length;
                }
            })
            .catch(error => console.log(error))
        axios_common.get('/sub3/likestories/', this.requestHeader)
            .then(response => {
                for (let i=0;i<response.data.length;i++){
                    this.like_stories.push(response.data[i].id)
                }
            })
            .catch(error => console.log(error))
        },
    }
</script>
<style scoped>
.bg {
    padding: 30px;
    background: rgb(244, 242, 243);
    /* background-size: cover; */
    /* background-repeat:no-repeat; */
    /* background-position: center; */
}
.font1 {font-family: 'Do Hyeon', sans-serif;}
#font2 {font-family: 'Black Han Sans', sans-serif;}
</style>
