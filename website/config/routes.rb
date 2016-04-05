Rails.application.routes.draw do

  get 'users/new'

  get 'search/index'

  get 'table/index'
  get 'table/display_report'
  get '/table_save_search' => "table#save_search"
  
  get 'graph/index'
  get '/performances' => "performances#index"
  
  get 'saved_search/index'
  get '/saved_search_destroy_reoccurence' => "saved_search#destroy_reoccurence"
  get '/saved_search_edit_reoccurence' => "saved_search#edit_reoccurence"
  get '/saved_search/editing_reoccurence' => "saved_search#editing_reoccurence"
  get 'teachings' => 'teachings#index'

  devise_for :users
  # The priority is based upon order of creation: first created -> highest priority.
  # See how all your routes lay out with "rake routes".

  # You can have the root of your site routed with "root"
  root 'problems#index'

  # Example of regular route:
  #   get 'products/:id' => 'catalog#view'

  # Example of named route that can be invoked with purchase_url(id: product.id)
  #   get 'products/:id/purchase' => 'catalog#purchase', as: :purchase

  # Example resource route (maps HTTP verbs to controller actions automatically):
  #   resources :products
  resources :table
  resources :saved_search
  resources :performances do
    collection do
      get 'report'
      get 'radar'
      get 'skillmeters'
      get 'list'
      get 'concept'
    end
  end

  resources :problems do
    collection do
      get 'user_select'
      get 'system_select'
      get 'review_list'
    end 
  end

  # Example resource route with options:
  #   resources :products do
  #     member do
  #       get 'short'
  #       post 'toggle'
  #     end
  #
  #     collection do
  #       get 'sold'
  #     end
  #   end

  # Example resource route with sub-resources:
  #   resources :products do
  #     resources :comments, :sales
  #     resource :seller
  #   end

  # Example resource route with more complex sub-resources:
  #   resources :products do
  #     resources :comments
  #     resources :sales do
  #       get 'recent', on: :collection
  #     end
  #   end

  # Example resource route with concerns:
  #   concern :toggleable do
  #     post 'toggle'
  #   end
  #   resources :posts, concerns: :toggleable
  #   resources :photos, concerns: :toggleable

  # Example resource route within a namespace:
  #   namespace :admin do
  #     # Directs /admin/products/* to Admin::ProductsController
  #     # (app/controllers/admin/products_controller.rb)
  #     resources :products
  #   end
end
